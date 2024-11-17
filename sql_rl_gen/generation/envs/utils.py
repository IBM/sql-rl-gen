import os
import re
from _csv import writer
from typing import Tuple, List, Dict, Any
import torch
from datasets import Dataset
from configs.config import WIKISQL_PATH
from data_preprocess.sql_utils import execute_query
from sql_rl_gen.feedback_metrics import calculate_all_feedback_metrics

def find_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.backends.cudnn.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def replace_columns(predicted_query, tables):
    for i in range(len(tables[0])):
        if tables[0][i] in predicted_query:
            predicted_query = predicted_query.replace(tables[0][i], tables[1][i])
        elif tables[0][i].upper() in predicted_query:
            predicted_query = predicted_query.replace(tables[0][i].upper(), tables[1][i])
    return predicted_query

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param

def extract_question(input_item: str) -> str:
    regex = r'question:\s*(.*)'
    code_string = re.search(regex, input_item, re.DOTALL)
    if code_string is not None and code_string.group(1) is not None:
        return code_string.group(1).strip()

def extract_data_to_llm(data) -> tuple[Any, Any, str, str, str, Any, Any]:
    if 'tables_sql' not in data:
        return str(data['prompt']).strip(), data['query'], str(data['tables']), str(data['db_id']), str(data['response']), None, None
    else:
        return str(data['prompt']).strip(), str(data['query']).strip(), str(data['tables']).replace("'", ''), str(data['db_id']), str(data['response']), data['tables_text'], data['tables_sql'],

def get_prompt(system, question, tables):
    prompt = f"""{system}. tables: {tables}. question: {question}"""
    return prompt

def prepare_observation_list_and_dataset_to_pass(dataset: Dataset) -> Tuple[List, Dict, Dict]:
    observation_list = []
    data_list_to_pass = {}
    columns_names_mismatch = {}
    for data in dataset:
        system, prompt, tables, db, expected_query, tables_text, tables_sql = extract_data_to_llm(data)
        observation_list.append({'input': f"{get_prompt(system, prompt, tables)}"})
        data_list_to_pass[prompt] = (db, expected_query)
        columns_names_mismatch[get_prompt(system, prompt, tables)] = (tables_text, tables_sql)
    return observation_list, data_list_to_pass, columns_names_mismatch

def sql_query_execution_feedback_on_dataset(dataset, dataset_path, input_item, predicted_text, columns_names_mismatch=None):
    if dataset_path == WIKISQL_PATH and predicted_text is not None and columns_names_mismatch is not None:
        predicted_text = replace_columns(predicted_text, columns_names_mismatch[input_item])
    question = extract_question(input_item)
    db_name, expected_query_from_db = dataset[question]
    return sql_query_execution_feedback(dataset_path, db_name, expected_query_from_db, predicted_text)

def sql_query_execution_feedback(dataset_path, db_id, expected_query, generated_query):
    if generated_query is None:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "error": True, "error_type": None, "error_description": None, "not_sql_format": True,
                "forbidden_sql_command": False, "output": generated_query, "expected": expected_query}
    if "drop" in generated_query.lower() or "delete" in generated_query.lower() or "insert" in generated_query.lower() or "create" in generated_query.lower():
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "error": True, "error_type": "Parsing_Exception", "error_description": None,
                "forbidden_sql_command": True, "not_sql_format": False, "output": generated_query, "expected": expected_query}
    expected_response_query = execute_query(dataset_path, db_id, expected_query)
    response_from_predicted_query = execute_query(dataset_path, db_id, generated_query)
    feedback = {}
    if isinstance(expected_response_query, Exception):
        feedback["error_type"] = type(response_from_predicted_query)
        feedback["error_reason"] = str(response_from_predicted_query)
        feedback["accuracy"] = 0.0
        feedback["precision"] = 0.0
        feedback["recall"] = 0.0
        feedback["f1"] = 0.0
        feedback["iou"] = 0.0
        feedback["not_sql_format"] = False
        feedback["forbidden_sql_command"] = False
    if isinstance(response_from_predicted_query, Exception):
        feedback["error_type"] = type(response_from_predicted_query)
        feedback["error_reason"] = str(response_from_predicted_query)
        feedback["accuracy"] = 0.0
        feedback["precision"] = 0.0
        feedback["recall"] = 0.0
        feedback["f1"] = 0.0
        feedback["iou"] = 0.0
        feedback["not_sql_format"] = False
        feedback["forbidden_sql_command"] = False
    else:
        feedback = calculate_all_feedback_metrics(expected_response_query, response_from_predicted_query)
    feedback["output"] = generated_query,
    feedback["expected"] = expected_query
    return feedback

def save_dict_csv(data_dict, file_path, file_name):
    file_dir = os.path.join(file_path, file_name)
    is_csv_file_created = os.path.isfile(file_dir)
    with open(file_dir, 'w' if not is_csv_file_created else 'a', newline='') as f:
        csv_writer = writer(f)
        if not is_csv_file_created:
            csv_writer.writerow(data_dict.keys())
        csv_writer.writerow(data_dict.values())