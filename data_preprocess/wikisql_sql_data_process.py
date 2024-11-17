import os.path
import sqlite3

from configs.config import DATA_PATH


def convert_row_data_to_lowercase(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        text_columns = [col[1] for col in columns if col[2].upper() == 'TEXT']
        if not text_columns:
            continue
        cursor.execute(f"SELECT rowid, * FROM {table_name};")
        rows = cursor.fetchall()
        for row in rows:
            rowid = row[0]
            original_values = row[1:]
            updated_values = []
            has_update = False
            for i, col_name in enumerate(text_columns):
                value = original_values[i]
                if isinstance(value, str):
                    lower_value = value.lower()
                    if lower_value != value:
                        has_update = True
                    updated_values.append(lower_value)
                else:
                    updated_values.append(value)
            if has_update:
                set_clause = ', '.join(f'"{col}" = ?' for col in text_columns)
                update_query = f"UPDATE {table_name} SET {set_clause} WHERE rowid = ?;"
                cursor.execute(update_query, (*updated_values, rowid))
    conn.commit()
    conn.close()


import json
import re
import records
from babel.numbers import parse_decimal, NumberFormatError

schema_re = re.compile(r'\((.+)\)')
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')
agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']

class DBEngine:
    def __init__(self, db_path):
        self.db = records.Database(f'sqlite:///{db_path}')

    def load_query_schema(self, table_id, select_index, agg_index, conditions):
        if not table_id.startswith('table'):
            table_id = f'table_{table_id.replace("-", "_")}'
        with self.db.get_connection() as conn:
            table_info = conn.query( f'SELECT sql FROM sqlite_master WHERE tbl_name = "{table_id}"' ).all()[0][0]
            schema_str = schema_re.findall(table_info.replace('\n', ''))[0]
        schema = {col.split()[0]: col.split()[1] for col in schema_str.split(', ')}
        select_clause = f"col{select_index}"
        if agg_ops[agg_index]:
            select_clause = f"{agg_ops[agg_index]}({select_clause})"
        where_clauses = []
        for col_index, op, val in conditions:
            if schema[f'col{col_index}'] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(val))
                except NumberFormatError:
                    val = float(num_re.findall(val)[0])
            where_str, val = string_to_format_sql(val)
            where_clauses.append(where_str.format(col_index, cond_ops[op], val))
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ''
        query = f"SELECT {select_clause} AS result FROM {table_id} {where_clause}"
        return query, schema

def string_to_format_sql(s):
    if isinstance(s, str):
        if '"' not in s:
            return 'col{} {} "{}"', s.lower()
        elif '"' in s and "'" not in s:
            return "col{} {} '{}'", s.lower()
        else:
            val = s.replace("'", "''").replace('"', '""')
            return 'col{} {} "{}"', val.lower()
    return 'col{} {} {}', s

def preprocess_tables(header_tokens):
    return ','.join(header_tokens).replace("'", "").replace('[', '').replace(']', '')

def extract_and_format(sql_data, table_data, db_engine):
    output = []
    for entry in sql_data:
        query, schema = db_engine.load_query_schema(
            entry['table_id'],
            entry['sql']['sel'],
            entry['sql']['agg'],
            entry['sql']['conds']
        )
        table_info = table_data[entry['table_id']]
        header_tokens = ['_'.join(header) for header in table_info['header_tok']]  # 1D array
        db_entry = {
            "db_id": "train",
            "instruction": "convert question and table into sql query",
            "tables_sql": list(schema.keys()),
            "tables_text": header_tokens,
            "tables": f"table_{entry['table_id'].replace('-', '_')}({preprocess_tables(header_tokens)})",
            "input": entry['question'].lower(),
            "output": query
        }
        output.append(db_entry)
    return output

def load_data(sql_path, table_path):
    with open(sql_path, 'r') as f:
        sql_data = [json.loads(line.strip()) for line in f]
    with open(table_path, 'r') as f:
        table_data = {entry['id']: entry for entry in (json.loads(line.strip()) for line in f)}
    return sql_data, table_data

def save_to_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    convert_row_data_to_lowercase(os.path.join(DATA_PATH, "wikisql/database/tables.sqlite"))

    sql_file_path = os.path.join(DATA_PATH, 'wikisql', 'dev.jsonl')
    table_file_path = os.path.join(DATA_PATH, 'wikisql', 'tables.jsonl')
    db_path = os.path.join(DATA_PATH, 'wikisql', 'database', 'tables.sqlite')
    sql_data, table_data = load_data(sql_file_path, table_file_path)
    db_engine = DBEngine(db_path)
    formatted_data = extract_and_format(sql_data, table_data, db_engine)
    save_to_json(formatted_data, os.path.join(DATA_PATH, 'example_text2sql_wikiSQL_dev.json'))

    sql_file_path = os.path.join(DATA_PATH, 'wikisql', 'train.jsonl')
    sql_data, table_data = load_data(sql_file_path, table_file_path)
    formatted_data = extract_and_format(sql_data, table_data, db_engine)
    save_to_json(formatted_data, os.path.join(DATA_PATH, 'example_text2sql_wikiSQL_train.json'))

    sql_file_path = os.path.join(DATA_PATH, 'wikisql', 'test.jsonl')
    sql_data, table_data = load_data(sql_file_path, table_file_path)
    formatted_data = extract_and_format(sql_data, table_data, db_engine)
    save_to_json(formatted_data, os.path.join(DATA_PATH, 'example_text2sql_wikiSQL_test.json'))

if __name__ == "__main__":
    main()
