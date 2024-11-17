import os
import json
import jsonlines
import sys
import re
import argparse
from tqdm import tqdm
from configs.config import INSTRUCTION_PROMPT, INSTRUCTION_ONE_SHOT_PROMPT, INPUT_PROMPT, SQL_DATA_INFO, \
    DATA_PATH

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

class ProcessSqlData:
    def __init__(
        self, train_file=None, dev_file=None, num_shot=0, code_representation=False
    ) -> None:
        self.train_file = train_file
        self.dev_file = dev_file
        self.num_shot = num_shot
        self.code_representation = code_representation

    def decode_json_file(
        self,
        data_file_list,
        table_file,
        db_folder_path,
        db_id_name,
        output_name,
        is_multiple_turn=False,
    ):
        if table_file.endswith(".jsonl"):
            tables = jsonlines.open(table_file)
            datas = []
            for data_file in data_file_list:
                datas.extend(jsonlines.open(data_file))
        elif table_file.endswith(".json"):
            tables = json.load(open(table_file))
            datas = []
            for data_file in data_file_list:
                datas.extend(json.load(open(data_file)))
        else:
            print("Unsupported file types")
            raise
        db_dict = {}
        for item in tables:
            tables = item["table_names_original"]
            coloumns = item["column_names_original"][1:]
            source = ""
            for i, name in enumerate(tables):
                data = [f'"{coloumn[1]}"' for coloumn in coloumns if coloumn[0] == i]
                source += (
                    name + "("
                )
                for d in data:
                    source += d
                    source += ","
                source = source[:-1]
                source += "), "
            source = source[:-2]
            db_dict[item["db_id"]] = source
        res = []
        base_instruction = INSTRUCTION_PROMPT
        if self.num_shot == 1:
            base_instruction = INSTRUCTION_ONE_SHOT_PROMPT
        for data in tqdm(datas):
            if data[db_id_name] in db_dict.keys():
                if is_multiple_turn:
                    history = []
                    for interaction in data["interaction"]:
                        input = {
                            "db_id": data[db_id_name],
                            "instruction": base_instruction,
                            "tables": db_dict[data[db_id_name]],
                            "question": INPUT_PROMPT.format(interaction["utterance"]),
                            "output": interaction[output_name],
                            "history": history,
                        }
                        res.append(input)
                        history.append(
                            (
                                INPUT_PROMPT.format(interaction["utterance"]),
                                interaction[output_name],
                            )
                        )
                else:
                    if self.code_representation:
                        db_path = os.path.join(db_folder_path, data[db_id_name])
                        sql_file_path = next(
                            (
                                file
                                for file in os.listdir(db_path)
                                if file.endswith(".sql")
                            ),
                            None,
                        )
                        if sql_file_path is None:
                            continue
                        schema_file_path = os.path.join(db_path, sql_file_path)
                        with open(schema_file_path, "r") as file:
                            schema_content = file.read()
                        create_statements = re.findall(
                            r"CREATE\s.*?;", schema_content, re.DOTALL|re.IGNORECASE
                        )
                        input = {
                            "db_id": data[db_id_name],
                            "instruction": create_statements,
                            "tables": db_dict[data[db_id_name]],
                            "question": INPUT_PROMPT.format(data["question"]),
                            "output": data[output_name],
                            "history": []
                        }
                        res.append(input)
                    else:
                        input = {
                            "db_id": data[db_id_name],
                            "instruction": base_instruction,
                            "tables": db_dict[data[db_id_name]],
                            "input": INPUT_PROMPT.format(data["question"]),
                            "output": data[output_name],
                            "history": [],
                        }
                        res.append(input)
        return res

    def create_sft_raw_data(self):
        train_data = []
        dev_data = []
        for data_info in SQL_DATA_INFO:
            train_data_file_list = [
                os.path.join(DATA_PATH, data_info["data_source"], file)
                for file in data_info["train_file"]
            ]
            train_data.extend(
                self.decode_json_file(
                    data_file_list=train_data_file_list,
                    table_file=os.path.join(
                        DATA_PATH,
                        data_info["data_source"],
                        data_info["train_tables_file"],
                    ),
                    db_folder_path=os.path.join(
                        DATA_PATH,
                        data_info["data_source"],
                        "database",
                    ),
                    db_id_name=data_info["db_id_name"],
                    output_name=data_info["output_name"],
                    is_multiple_turn=data_info["is_multiple_turn"],
                )
            )
            dev_data_file_list = [
                os.path.join(DATA_PATH, data_info["data_source"], file)
                for file in data_info["dev_file"]
            ]
            dev_data.extend(
                self.decode_json_file(
                    data_file_list=dev_data_file_list,
                    table_file=os.path.join(
                        DATA_PATH,
                        data_info["data_source"],
                        data_info["dev_tables_file"],
                    ),
                    db_folder_path=os.path.join(
                        DATA_PATH,
                        data_info["data_source"],
                        "database",
                    ),
                    db_id_name=data_info["db_id_name"],
                    output_name=data_info["output_name"],
                    is_multiple_turn=data_info["is_multiple_turn"],
                )
            )
        with open(self.train_file, "w", encoding="utf-8") as s:
            json.dump(train_data, s, indent=4, ensure_ascii=False)
        with open(self.dev_file, "w", encoding="utf-8") as s:
            json.dump(dev_data, s, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code_representation", help="Enable code representation", default=False
    )
    parser.add_argument(
        "--dataset", help="Enable code representation", default="spider"
    )
    args = parser.parse_args()
    all_in_one_train_file = os.path.join(DATA_PATH, f"example_text2sql_{args.dataset}_train.json")
    all_in_one_dev_file = os.path.join(DATA_PATH, f"example_text2sql_{args.dataset}_dev.json")
    precess = ProcessSqlData(
        train_file=all_in_one_train_file,
        dev_file=all_in_one_dev_file,
        code_representation=args.code_representation,
    )
    precess.create_sft_raw_data()
    one_shot_all_in_one_train_file = os.path.join(
        DATA_PATH, f"example_text2sql_{args.dataset}_train_one_shot.json"
    )
    one_shot_all_in_one_dev_file = os.path.join(
        DATA_PATH, f"example_text2sql_{args.dataset}_dev_one_shot.json"
    )
    one_shot_precess = ProcessSqlData(
        train_file=one_shot_all_in_one_train_file,
        dev_file=one_shot_all_in_one_dev_file,
        num_shot=1,
        code_representation=args.code_representation,
    )
    one_shot_precess.create_sft_raw_data()
