import os

import duckdb
from dotenv import load_dotenv

load_dotenv()

PROJECT_HOME = os.getenv('TOXIC_X_DOM_HOME')
RESULTS_FILE = 'results.duckdb'


def open_db(file_name=RESULTS_FILE):
    file_path = os.path.join(PROJECT_HOME, file_name)
    return duckdb.connect(database=file_path, read_only=False)


def initialize_db(file_name=RESULTS_FILE):

    con = open_db(file_name)

    with open(os.path.join(PROJECT_HOME, 'src/toxic_x_dom/schema.sql'), 'r') as schema_file:
        query = schema_file.read()
        con.execute(query)

    # TESTING
    con.execute(
        "INSERT INTO evaluation "
        "VALUES (DEFAUlT, 'test123', 'train123', FALSE, -1, NULL, NULL, NULL, 0.4, NULL, NULL, NULL, NULL, NULL, 0.6, NULL, NULL) "
        "RETURNING id; "
    )
    uuids = con.fetchall()

    con.close()


if __name__ == "__main__":
    initialize_db()
