import os
import sys

from transformers import TrainingArguments, HfArgumentParser

from toxic_x_dom.rationale_extraction.search import RationaleExtractionArguments
from toxic_x_dom.span_prediction.huggingface import ModelArguments, DataTrainingArguments
from toxic_x_dom.span_prediction.search import eval_trials
from toxic_x_dom.results_db import open_db
from toxic_x_dom.span_prediction.search import extract_train_dataset_and_model_keys


def eval_on_test(model_args, data_args, train_args):
    db = open_db()

    train_dataset, model_key = extract_train_dataset_and_model_keys(model_args)

    # get best performing trials
    columns = ",".join(['eval_dataset', 'train_dataset', 'propagate_binary', 'filling_chars',
                        'model_key'])
    results = db.query(f"SELECT {columns} "
                       f"FROM tuned_in_domain_max_f1_har_macro_view "
                       f"WHERE method_type = 'span_pred'"
                       f"AND train_dataset = '{train_dataset}'"
                       f"AND model_key = '{model_key}'"
                       f"AND NOT CONTAINS(eval_dataset, 'test')").df()

    print('Running eval on test split for best trials:')
    print(results)

    # construct test configs
    configs = []
    for _, row in results.iterrows():
        config = (row.eval_dataset, row.filling_chars, row.propagate_binary)
        configs.append(config)

    data_args.eval_split = 'test'
    eval_trials(model_args, data_args, train_args, configs)


if __name__ == "__main__":
    parser = HfArgumentParser((RationaleExtractionArguments, ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        _, _model_args, _data_args, _train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        _, _model_args, _data_args, _train_args = parser.parse_args_into_dataclasses()

    eval_on_test(_model_args, _data_args, _train_args)
