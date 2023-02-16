import os
import uuid

import duckdb
import pandas as pd
from dotenv import load_dotenv

from toxic_x_dom.data import load_toxic_span_datasets

load_dotenv()

PROJECT_HOME = os.getenv('TOXIC_X_DOM_HOME')
RESULTS_FILE = 'results.duckdb'

EVALUATION_COLUMNS = ['id', 'eval_dataset', 'train_dataset', 'propagate_binary', 'filling_chars', 'f1_micro',
                      'precision_micro', 'recall_micro', 'f1_toxic', 'precision_toxic', 'recall_toxic',
                      'f1_toxic_no_span', 'precision_toxic_no_span', 'recall_toxic_no_span', 'f1_non_toxic',
                      'precision_non_toxic', 'recall_non_toxic', 'non_toxic_pct_predicted', 'nr_empty_pred',
                      'nr_empty_label', 'nr_empty_both', 'nr_samples']

SPAN_DATA_COLUMNS = ['sample_id', 'dataset_key', 'split', 'full_text', 'toxic', 'toxic_mask', 'toxic_tokens']

PREDICTIONS_COLUMNS = ['evaluation_id', 'sample_id', 'toxic_prediction', 'span_prediction']


def open_db(file_name=RESULTS_FILE):
    file_path = os.path.join(PROJECT_HOME, file_name)
    return duckdb.connect(database=file_path, read_only=False)


def insert_evaluation(df):
    """
    df: DataFrame with a row for each evaluation that is to be inserted
    """
    df['id'] = [uuid.uuid4() for _ in range(len(df.index))]
    db = open_db()

    columns = ','.join(EVALUATION_COLUMNS)
    db.execute(f'INSERT INTO evaluation({columns}) SELECT {columns} FROM df;')

    db.close()
    return df


def insert_predictions(evaluation_ids, evaluation_predictions):

    for eval_id, eval_predictions in zip(evaluation_ids, evaluation_predictions):
        eval_predictions['evaluation_id'] = eval_id
    all_predictions = pd.concat(evaluation_predictions)

    db = open_db()

    columns = ','.join(PREDICTIONS_COLUMNS)
    db.execute(f'INSERT INTO predictions({columns}) SELECT {columns} FROM all_predictions')

    db.close()


def initialize_db(file_name=RESULTS_FILE):
    con = open_db(file_name)

    with open(os.path.join(PROJECT_HOME, 'src/toxic_x_dom/schema.sql'), 'r') as schema_file:
        query = schema_file.read()
        con.execute(query)

    datasets = load_toxic_span_datasets()

    columns = ','.join(SPAN_DATA_COLUMNS)
    for key, dataset in datasets.items():
        dataset['sample_id'] = dataset['id']
        dataset['dataset_key'] = key
        con.execute(f'INSERT INTO span_data({columns}) SELECT {columns} FROM dataset')

    con.close()


if __name__ == "__main__":
    initialize_db()
