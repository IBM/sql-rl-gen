import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT_PATH, "sql-rl-gen/data_preprocess/data")
SPIDER_PATH = os.path.join(DATA_PATH, "spider")
SPIDER_DATABASES_PATH = os.path.join(SPIDER_PATH, "database")
BIRD_PATH = os.path.join(DATA_PATH, "bird")
BIRD_DATABASES_TRAIN_PATH = os.path.join(BIRD_PATH, "train", "train_databases")
BIRD_DATABASES_DEV_PATH = os.path.join(BIRD_PATH, "dev", "dev_databases")
WIKISQL_PATH = os.path.join(DATA_PATH, "wikiSQL")
EXT2TYPE = {"csv": "csv", "json": "json", "jsonl": "json", "txt": "text"}
OUT_DIR = os.path.join(ROOT_PATH, "sql-rl-gen/output/")
OUT_STATISTICS_DIR = os.path.join(OUT_DIR, "statistics")
SQL_DATA_INFO = [
    {
        "data_source": "spider", #"wikisql",
        "train_file":  ["train_spider.json", "train_others.json"],
        "dev_file": ["dev.json"],
        "train_tables_file": "tables.json",
        "dev_tables_file": "tables.json",
        "db_id_name": "db_id",
        "output_name": "query",
        "is_multiple_turn": False,
    }
]
INSTRUCTION_PROMPT = "convert question and table into SQL query."
INPUT_PROMPT = "{}"
INSTRUCTION_ONE_SHOT_PROMPT = """\
I want you to act as a SQL terminal in front of an example database. \
You need only to return the sql command to me. \
First, I will show you few examples of an instruction followed by the correct SQL response. \
Then, I will give you a new instruction, and you should write the SQL response that appropriately completes the request.\
\n### Example1 Instruction:
The database contains tables such as employee, salary, and position. \
Table employee has columns such as employee_id, name, age, and position_id. employee_id is the primary key. \
Table salary has columns such as employee_id, amount, and date. employee_id is the primary key. \
Table position has columns such as position_id, title, and department. position_id is the primary key. \
The employee_id of salary is the foreign key of employee_id of employee. \
The position_id of employee is the foreign key of position_id of position.\
\n### Example1 Input:\nList the names and ages of employees in the 'Engineering' department.\n\
\n### Example1 Response:\nSELECT employee.name, employee.age FROM employee JOIN position ON employee.position_id = position.position_id WHERE position.department = 'Engineering';\
\n###New Instruction:\n{}\n"""
