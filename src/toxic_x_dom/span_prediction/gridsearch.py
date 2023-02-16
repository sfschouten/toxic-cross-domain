import os
import sys
from dataclasses import field, dataclass
from typing import List

import pandas as pd
from transformers import HfArgumentParser, TrainingArguments

from toxic_x_dom.data import SPAN_DATASETS
from toxic_x_dom.results_db import insert_evaluation, open_db
from toxic_x_dom.span_prediction.huggingface import ModelArguments, DataTrainingArguments, EvaluationArguments, main


@dataclass
class GridSearchArguments:

    propagate_binary_predictions: List[bool] = field(
        default_factory=lambda: [True, False], metadata={"help": ""}
    )

    filling_chars: List[int] = field(
        default_factory=lambda: [0, 1, 9999], metadata={"help": ""}
    )


def gridsearch(grid_args, model_args, data_args, train_args):
    train_dataset_name = model_args.model_name_or_path.split('-')[-1].replace('/', '')

    data_args.include_nontoxic_samples = True

    results = []
    for eval_dataset in SPAN_DATASETS.keys():
        data_args.dataset_name = eval_dataset
        for filling_chars in grid_args.filling_chars:
            for binary_propagate in grid_args.propagate_binary_predictions:
                eval_args = EvaluationArguments()
                eval_args.filling_chars = filling_chars
                eval_args.binary_propagate = binary_propagate

                result_dict = main(model_args, data_args, train_args, eval_args)
                metrics = {key.replace("eval_", ""): value for key, value in result_dict['eval_metrics'].items()}

                results.append(metrics | {
                    "train_dataset": train_dataset_name,
                    "eval_dataset": eval_dataset,
                    "propagate_binary": binary_propagate,
                    "filling_chars": filling_chars,
                })

    results_df = pd.DataFrame(results)
    results_df = insert_evaluation(results_df)

    db = open_db()

    PREDICTIONS_COLUMNS = ['id']
    columns = ','.join(PREDICTIONS_COLUMNS)
    db.execute(f'INSERT INTO prediction_evaluation({columns}) SELECT {columns} FROM results_df;')

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

    gridsearch(_attr_args, _model_args, _data_args, _train_args)
