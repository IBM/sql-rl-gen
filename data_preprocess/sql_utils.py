import re
import sqlite3
import os
from typing import Optional

def create_connection(path, db_file):
    if not os.path.exists(path):
        raise Exception("Spider database does not exist! Download it from: https://github.com/eosphoros-ai/DB-GPT-Hub?tab=readme-ov-file#21-dataset")
    db_path = os.path.join(path, db_file, f"{db_file}.sqlite")
    if not os.path.exists(db_path):
        raise Exception(f"Spider database does not have {db_file} schema! Download it from: https://github.com/eosphoros-ai/DB-GPT-Hub?tab=readme-ov-file#21-dataset")
    return sqlite3.connect(db_path)

def execute_query(dataset_path, db_name, query):
    connection = create_connection(dataset_path, db_name)
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchall()
    except Exception as e:
        connection.close()
        return e
    connection.close()
    return result

def get_generated_sql_query_from_response(response: str) -> Optional[str]:
    regex_patterns = [r'```(?:sql|SQL)?\s*(.*?)(?:```|;$)', r'(SELECT\s.*?;)[\'"]?\s*$']
    for pattern in regex_patterns:
        code_string = re.search(pattern, response, re.DOTALL)
        if code_string is not None and code_string.group(1) is not None:
            return code_string.group(1).strip()
