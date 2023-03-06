import os
import sys

from transformers import HfArgumentParser

from toxic_x_dom.binary_classification.huggingface import ModelArguments, DataTrainingArguments, TrainingArguments
from toxic_x_dom.rationale_extraction.search import eval_trials, RationaleExtractionArguments, extract_train_dataset_key
from toxic_x_dom.results_db import open_db


def eval_on_test(model_args, data_args, train_args):
    db = open_db()

    train_dataset = extract_train_dataset_key(model_args)

    # get best performing trials
    columns = ",".join(['eval_dataset', 'train_dataset', 'propagate_binary', 'filling_chars', 'attribution_method',
                        'scale_scores', 'cumulative_scoring', 'threshold'])
    results = db.query(f"SELECT {columns} FROM trial_evaluations_to_test"
                       f" WHERE method_type = 'rationale'"
                       f" AND train_dataset = '{train_dataset}'").df()

    print('Running eval on test split for best trials:')
    print(results)

    # construct test configs
    configs = []
    for _, row in results.iterrows():
        config_p1 = (row.eval_dataset, f'toxic_x_dom.rationale_extraction.attribution.{row.attribution_method}')
        config_p2 = [(row.scale_scores, row.cumulative_scoring, row.propagate_binary, row.threshold, row.filling_chars)]
        configs.append((config_p1, config_p2))

    data_args.attribution_split = 'test'
    if len(configs) > 0:
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
