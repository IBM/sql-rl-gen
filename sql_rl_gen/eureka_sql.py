import ast
import datetime
import json
import logging
import os
import re
import shutil
import sys
import time
from typing import Dict, List
import hydra
import numpy as np
import ollama
import subprocess as sp
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
EUREKA_ROOT_DIR = os.path.join(os.getcwd(), 'sql-rl-gen')
DESCRIPTIONS_DIR = f'{EUREKA_ROOT_DIR}/descriptions'
RL_ROOT_DIR = f'{EUREKA_ROOT_DIR}/generation'
ENV_DIR = f'{RL_ROOT_DIR}/envs'
output_file = f'{ENV_DIR}/sql_generation_environment.py'

def get_free_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    time_start = datetime.datetime.now()
    while True:
        rl_log = file_to_string(rl_filepath)
        if "finish" in rl_log or "Traceback" in rl_log or "File" in rl_log or time_start + datetime.timedelta(
                minutes=30) < datetime.datetime.now():
            if log_status and "finish" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            elif log_status and ("Traceback" in rl_log or "File" in rl_log):
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"The agent stucked in iteration {iter_num}: Code Run is {response_id} ")
            break

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def get_function_signature(code_string):
    module = ast.parse(code_string)
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if not function_defs:
        return None
    function_def = function_defs[0]
    input_lst = []
    signature = function_def.name + '(' + ', '.join(arg.arg for arg in function_def.args.args) + ')'
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return signature, input_lst

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def load_tensorboard_logs(path):
    data = pd.read_csv(path, delim_whitespace=True)
    data_dict = data.to_dict(orient='list')
    return data_dict

def extract_evaluation_episodes(line_content, rewards_list):
    pattern = r"Reward: ([-+]?[0-9]*\.?[0-9]+)"
    matches = re.findall(pattern, line_content)
    for match in matches:
        rewards_list.append(float(match[3]))

def extract_stats(line_content, statistics_list: Dict[str, List]):
    match = re.search(r"Reward:\s*(-?[\d.]+),\s*Metrics:\s*\{([^}]*)\}", line_content)
    reward = float(match.group(1))
    metrics_content = match.group(2)
    metrics = re.findall(r"'([^']+)':\s*([^,]+)", metrics_content)
    for key, value in metrics:
        if key in statistics_list:
            statistics_list[key].append(float(value) if bool(re.match(r'^[-+]?[0-9]*\.?[0-9]+$', value)) else value)
        else:
            statistics_list[key] = [float(value) if bool(re.match(r'^[-+]?[0-9]*\.?[0-9]+$', value)) else value]
    if 'reward' in statistics_list:
        statistics_list['reward'].append(reward)
    else:
        statistics_list['reward'] = [reward]

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")
    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)
    env_name = cfg.env.env_name.lower()
    task_file = f'{ENV_DIR}/{env_name}_environment_obs.py'
    task_obs_file = f'{ENV_DIR}/{env_name}_environment_obs.py'
    shutil.copy(task_obs_file, f"env_init_environment.py")
    task_code_string = file_to_string(task_file)
    task_obs_code_string = file_to_string(task_obs_file)
    initial_system = file_to_string(f'{DESCRIPTIONS_DIR}/initial_system.txt')
    initial_user = file_to_string(f'{DESCRIPTIONS_DIR}/initial_user.txt')
    reward_signature = file_to_string(f'{DESCRIPTIONS_DIR}/reward_signature.txt')
    policy_feedback = file_to_string(f'{DESCRIPTIONS_DIR}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{DESCRIPTIONS_DIR}/execution_error_feedback.txt')
    calculate_feedback_tip = file_to_string(f'{DESCRIPTIONS_DIR}/feedback_func_tip.txt')
    code_output_tip = file_to_string(f'{DESCRIPTIONS_DIR}/code_output_tip.txt')
    initial_system = initial_system.format(task_reward_signature_string=reward_signature)
    initial_user = initial_user.format(task_environment_code_string=task_obs_code_string, task_description=task_description)
    initial_user += ("\n" + calculate_feedback_tip)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]
    task_code_string = task_code_string.replace(task, task + suffix)
    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None
    for iter in range(cfg.iteration):
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4
        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")
        while True:
            if total_samples >= cfg.sample:
                break
            for response_id in range(cfg.sample):
                for attempt in range(1000):
                    try:
                        response_cur = ollama.chat(model=model, messages=messages, stream=False, options={"temperature": cfg.temperature, "num_ctx": 15000})
                        if not response_cur["done"]:
                            raise Exception("Non-200 response: " + str(response_cur))
                        total_samples += chunk_size
                        break
                    except Exception as e:
                        if attempt >= 10:
                            chunk_size = max(int(chunk_size / 2), 1)
                            print("Current Chunk Size", chunk_size)
                        logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                        time.sleep(1)
                if response_cur is None:
                    logging.info("Code terminated due to too many failed attempts!")
                    exit()
                responses.append(str(response_cur["message"]["content"]))
                total_completion_token += response_cur["eval_count"]
                total_token += total_completion_token
        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0] + "\n")
        logging.info(f"Iteration {iter}: Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        code_runs = []
        rl_runs = []
        for response_id in range(cfg.sample):
            response_cur = responses[response_id]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")
            patterns = [r'```python(.*?)```', r'```(.*?)```', r'"""(.*?)"""', r'""(.*?)""', r'"(.*?)"']
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
            try:
                reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                continue
            code_runs.append(code_string)
            indent = ' ' * 8
            reward_signature = indent + "return " + reward_signature
            if "def compute_reward(self, input_item, predicted_text)" in task_code_string:
                task_code_string_iter = task_code_string.replace(
                    "def compute_reward(self, input_item, predicted_text) -> Tuple[float, Dict]",
                    "def compute_reward(self, input_item, predicted_text) -> Tuple[float, Dict]:\n" + reward_signature)
            else:
                raise NotImplementedError
            with open(output_file, 'w') as file:
                file.writelines(task_code_string_iter + '\n')
                file.writelines(code_string + '\n')
            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')
            shutil.copy(task_obs_file, f"env_iter{iter}_response{response_id}.py")
            while get_free_gpu_memory()[0] < 2000:
                time.sleep(10)
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                process = sp.Popen(['python', '-u', f'{RL_ROOT_DIR}/sql_generation.py', '--model_name_or_path', "juierror/flan-t5-text2sql-with-schema-v2", '--dataset', "example_text2sql_train",
                                    '--steps_n', '200', '--template', 'llama2', '--dataset_name', 'spider', '--outdir', f'{iter}_{response_id}'], stdout=f, stderr=f)
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)
        if len(code_runs) == 0:
            continue
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []
        exec_success = False
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read()
            except:
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                contents.append(content)
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue
            content = ''
            traceback_msg = filter_traceback(stdout_str)
            if traceback_msg == '':
                exec_success = True
                lines = stdout_str.split('\n')
                stats = {}
                content += policy_feedback
                for i, line in enumerate(lines):
                    if line.startswith('Reward:'):
                        extract_stats(line, stats)
                content += str(stats)
                try:
                    average_accuracy = sum(stats['accuracy']) / len(stats['accuracy']) if stats['accuracy'] else 0.0
                    successes.append(average_accuracy)
                except Exception as e:
                    successes.append(DUMMY_FAILURE)
                    reward_correlations.append(DUMMY_FAILURE)
                    content += execution_error_feedback.format(traceback_msg="The statistics is not full! It lacks some fields!")
                content += "\n" + code_output_tip
            else:
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)
            contents.append(content)
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]
        max_success = successes[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_reward_code_path = code_paths[best_sample_idx]
        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        best_code_paths.append(code_paths[best_sample_idx])
        logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        try:
            logging.info(f"Iteration {iter}: LLM Output Content:\n" + responses[best_sample_idx]["message"]["content"] + "\n")
        except Exception as e:
            logging.info('Parsing error')
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{cfg.env.task}')
        x_axis = np.arange(len(max_successes))
        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")
        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")
        fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')
        np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)
        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx]}]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]}
            messages[-1] = {"role": "user", "content": best_content}
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)
    if max_reward_code_path is None:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(
        f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    shutil.copy(max_reward_code_path, output_file)
    eval_runs = []
    for i in range(cfg.num_eval):
        while get_free_gpu_memory()[0] < 2000:
            time.sleep(10)
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'w') as f:
            process = sp.Popen(['python', '-u', f'{RL_ROOT_DIR}/sql_generation.py', '--model_name_or_path', "juierror/flan-t5-text2sql-with-schema-v2", '--dataset', "example_text2sql_train",
                                '--steps_n', '200', '--template', 'llama2', '--dataset_name', 'spider', '--outdir', f'final_best'], stdout=f, stderr=f)
        block_until_training(rl_filepath)
        eval_runs.append(process)
    reward_code_final_successes = []
    rl_filepath = f"reward_code_eval{i}.txt"
    with open(rl_filepath, 'r') as f:
        stdout_str = f.read()
    lines = stdout_str.split('\n')
    stats = {}
    for i, line in enumerate(lines):
        if line.startswith('Reward:'):
            extract_stats(line, stats)
    logging.info(f"Final feedback: {stats}")
    np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes)

if __name__ == "__main__":
    main()