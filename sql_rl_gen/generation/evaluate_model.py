import logging
import os.path
import sys
from os import listdir
from statistics import mode
import torch
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForSeq2SeqLM
from configs.config import BIRD_DATABASES_DEV_PATH, SPIDER_DATABASES_PATH, WIKISQL_PATH
from configs.data_args import DataArguments, DatasetName
from configs.rl_args import EvaluateArguments, EvaluationMethod
from data_preprocess.data_utils import get_dataset
from sklearn.model_selection import KFold
from sql_rl_gen.generation.envs.sql_generation_environment import SQLRLEnv
from sql_rl_gen.generation.envs.utils import find_device, sql_query_execution_feedback_on_dataset, prepare_observation_list_and_dataset_to_pass, save_dict_csv
from sql_rl_gen.generation.rllib.custom_actor import CustomActor
from sql_rl_gen.generation.rllib.custom_trainer import train_evaulate_agent

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = find_device()

def prediction_feedback(statistics, data_list_to_pass, dataset_path, observation, result, columns_names_mismatch):
    feedback = sql_query_execution_feedback_on_dataset(data_list_to_pass, dataset_path, observation['input'], result, columns_names_mismatch)
    for k, v in feedback.items():
        if k in statistics:
            statistics[k].append(v)
        else:
            statistics[k] = [v]
        logging.info(f"Appending to {k} value: {v}")
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
    elif device == torch.device("mps"):
        torch.mps.empty_cache()
    return feedback

def construct_file(statistics):
    feedback = {}
    for k, v in statistics.items():
        if isinstance(v[0], float):
            feedback[k] = {}
            feedback[k]['maximum'] = max(v)
            feedback[k]['mean'] = sum(v) / len(v)
            feedback[k]['minimum'] = min(v)
        else:
            mode_of_stat = mode(v)
            feedback[k] = {}
            feedback[k]['most_common'] = mode_of_stat
            feedback[k]['number_of_occurrence'] = len(list(filter(lambda x: x is mode_of_stat, v)))
    return feedback

def create_data_from_indexes(kfolds, observation_list):
    to_return_data = []
    for i in range(len(kfolds)):
        test_data = []
        train_data = []
        for j in range(len(kfolds[i][0])):
            train_data.append(observation_list[kfolds[i][0][j]])
        for j in range(len(kfolds[i][1])):
            test_data.append(observation_list[kfolds[i][1][j]])
        to_return_data.append((train_data, test_data))
    return to_return_data

def prepare_for_evaluate(eval_args: EvaluateArguments, data_args: DataArguments):
    dataset = get_dataset(data_args)
    dataset = dataset.take(eval_args.number_of_rows_to_use)
    if data_args.dataset_name == DatasetName.SPIDER.value:
        dataset_path = SPIDER_DATABASES_PATH
    elif data_args.dataset_name == DatasetName.WIKISQL.value:
        dataset_path = WIKISQL_PATH
    else:
        dataset_path = BIRD_DATABASES_DEV_PATH
    observation_list, data_list_to_pass, columns_names_mismatch = prepare_observation_list_and_dataset_to_pass(dataset)
    tokenizer = AutoTokenizer.from_pretrained(eval_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(eval_args.model_name_or_path)
    model.to(device)
    model.eval()
    os.makedirs(eval_args.outdir, exist_ok=True)
    return dataset_path, observation_list, data_list_to_pass, columns_names_mismatch, model, tokenizer

def evaluate_models(eval_args: EvaluateArguments, data_args: DataArguments):
    dataset_path, observation_list, data_list_to_pass, columns_names_mismatch, model, tokenizer = prepare_for_evaluate(eval_args, data_args)
    model.eval()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
    logger = logging.getLogger("Test")
    if eval_args.trained_agent_path:
        env = SQLRLEnv(model, tokenizer, data_list_to_pass, dataset_path, eval_args.outdir, logger, "evaluation", columns_names_mismatch=columns_names_mismatch,
                       observation_input=observation_list, compare_sample=1)
        actor = CustomActor(env, model, tokenizer, temperature=eval_args.temperature, top_k=eval_args.top_k, top_p=eval_args.top_p)
        agent = actor.agent_ppo(update_interval=eval_args.update_interval, minibatch_size=eval_args.minibatch_size, epochs=eval_args.epochs, lr=eval_args.lr)
        agent.load(eval_args.trained_agent_path)
    statistics = {}
    for observation in observation_list:
        if eval_args.trained_agent_path:
            result = actor.predict(observation)[0][:-4]
        else:
            input_ids = tokenizer(observation['input'], return_tensors="pt", max_length=1024).input_ids
            input_ids = input_ids.to(device)
            generated_ids = model.generate(input_ids, max_length=120, min_length=10)
            result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # GC is not working properly with video card. It's better to clean manually to prevent OutOfMemoryException
            del input_ids
            del generated_ids
        prediction_feedback(statistics, data_list_to_pass, dataset_path, observation, result, columns_names_mismatch)
    feedback = construct_file(statistics)
    save_dict_csv(feedback, eval_args.outdir, "feedback_metrics")
    save_dict_csv(statistics, eval_args.outdir, "statistics_metrics")
    return feedback, statistics

def cross_validation_evaluate(eval_args: EvaluateArguments, data_args: DataArguments):
    dataset_path, observation_list, data_list_to_pass, columns_names_mismatch, model, tokenizer = prepare_for_evaluate(eval_args, data_args)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
    logger = logging.getLogger("Cross Validation")
    ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results
    kfolds = list(kf.split(observation_list))
    kfolds_data = create_data_from_indexes(kfolds, observation_list)
    all_feedback = []
    all_statistics = []
    latest_checkpoint = f'../train/0_checkpoint'
    for i in range(len(kfolds_data)):
        model.train()
        # train folds before testing fold
        env = SQLRLEnv(model, tokenizer, data_list_to_pass, dataset_path, '../train/', logger, "training",
                       columns_names_mismatch=columns_names_mismatch, observation_input=observation_list, compare_sample=1)
        actor = CustomActor(env, model, tokenizer, temperature=eval_args.temperature, top_k=eval_args.top_k, top_p=eval_args.top_p)
        agent = actor.agent_ppo(update_interval=eval_args.update_interval, minibatch_size=eval_args.minibatch_size, epochs=eval_args.epochs, lr=eval_args.lr)
        agent.load(f'{latest_checkpoint}/827_finish' if i > 0 else eval_args.trained_agent_path)
        latest_checkpoint = f'../train/{i}_checkpoint'
        try:
            train_evaulate_agent(agent, env, steps=len(kfolds_data[i][0]), outdir=latest_checkpoint, eval_n_steps=None, eval_n_episodes=5, train_max_episode_len=1000, eval_interval=10)
        except Exception as e:
            print(e)
        # testing fold
        model.eval()
        env = SQLRLEnv(model, tokenizer, data_list_to_pass, dataset_path, '../test/', logger, "testing",
                       columns_names_mismatch=columns_names_mismatch, observation_input=observation_list, compare_sample=1)
        actor = CustomActor(env, model, tokenizer, temperature=eval_args.temperature, top_k=eval_args.top_k, top_p=eval_args.top_p)
        agent = actor.agent_ppo(update_interval=eval_args.update_interval, minibatch_size=eval_args.minibatch_size, epochs=eval_args.epochs, lr=eval_args.lr)
        filename = "best"
        list_directory = listdir(f"{latest_checkpoint}")
        for file_name in list_directory:
            if "finish" in file_name:
                filename = file_name
        agent.load(f'{latest_checkpoint}/{filename}')
        statistics = {}
        for observation in kfolds_data[i][1]:
            result = actor.predict(observation)[0][:-4]
            prediction_feedback(statistics, data_list_to_pass, dataset_path, observation, result, columns_names_mismatch)
        feedback = construct_file(statistics)
        all_feedback.append(feedback)
        all_statistics.append(statistics)
    save_dict_csv(all_feedback, eval_args.outdir, "feedback_metrics")
    save_dict_csv(all_statistics, eval_args.outdir, "statistics_metrics")
    return all_feedback, all_statistics

def compare_models_rl():
    parser = HfArgumentParser((EvaluateArguments, DataArguments))
    eval_args, data_args = parser.parse_args_into_dataclasses()
    data_args.init_for_training()
    if eval_args.evaluation_method == EvaluationMethod.KFOLD:
        cross_validation_evaluate(eval_args, data_args)
    else:
        evaluate_models(eval_args, data_args)

if __name__ == "__main__":
    compare_models_rl()
