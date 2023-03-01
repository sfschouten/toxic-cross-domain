import os
import sys
from dataclasses import field, dataclass
from typing import List

import pandas as pd
from transformers import HfArgumentParser, TrainingArguments

from toxic_x_dom.data import SPAN_DATASETS
from toxic_x_dom.results_db import insert_evaluation, open_db, insert_predictions
from toxic_x_dom.span_prediction.huggingface import ModelArguments, DataTrainingArguments, EvaluationArguments, main


@dataclass
class GridSearchArguments:

    propagate_binary_predictions: List[bool] = field(
        default_factory=lambda: [True, False], metadata={"help": ""}
    )

    filling_chars: List[int] = field(
        default_factory=lambda: [0, 1, 9999], metadata={"help": ""}
    )


def _generate_grid(grid_args):
    trial_configs = []
    for eval_dataset in SPAN_DATASETS.keys():
        for filling_chars in grid_args.filling_chars:
            for binary_propagate in grid_args.propagate_binary_predictions:
                trial_configs.append((eval_dataset, filling_chars, binary_propagate))
    return trial_configs


def extract_train_dataset_and_model_keys(model_args):
    model_str_parts = model_args.model_name_or_path.split('/')[-2].split('-')
    train_dataset_name = model_str_parts[-1]
    model_key = "-".join(model_str_parts[:-1])
    return train_dataset_name, model_key


def eval_trials(model_args, data_args, train_args, trial_configs):
    train_dataset_name, model_key = extract_train_dataset_and_model_keys(model_args)   

    # enforce the use of nontoxic examples so we get a complete evaluation
    data_args.include_nontoxic_samples = True

    results = []
    predictions = []

    for trial_config in trial_configs:
        eval_dataset, filling_chars, binary_propagate = trial_config

        data_args.dataset_name = eval_dataset

        eval_args = EvaluationArguments()
        eval_args.filling_chars = filling_chars
        eval_args.propagate_binary = binary_propagate

        result_dict = main(model_args, data_args, train_args, eval_args)
        metrics = {key.replace("eval_", ""): value for key, value in result_dict['eval_metrics'].items()}

        results.append(metrics | {
            "train_dataset": train_dataset_name,
            "eval_dataset": f"{eval_dataset}-{data_args.eval_split}",
            "propagate_binary": binary_propagate,
            "filling_chars": filling_chars,
        })
        predictions.append(result_dict['predictions'])

    results_df = pd.DataFrame(results)
    results_df['model_key'] = model_key
    results_df = insert_evaluation(results_df)

    insert_predictions(results_df['id'], predictions)

    db = open_db()

    PREDICTIONS_COLUMNS = ['id', 'model_key']
    columns = ','.join(PREDICTIONS_COLUMNS)
    db.execute(f'INSERT INTO span_pred_evaluation({columns}) SELECT {columns} FROM results_df;')

    db.close()


if __name__ == "__main__":
    parser = HfArgumentParser(
        (GridSearchArguments, ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        _attr_args, _model_args, _data_args, _train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        _attr_args, _model_args, _data_args, _train_args = parser.parse_args_into_dataclasses()

    _trial_configs = _generate_grid(_attr_args)
    eval_trials(_model_args, _data_args, _train_args, _trial_configs)
