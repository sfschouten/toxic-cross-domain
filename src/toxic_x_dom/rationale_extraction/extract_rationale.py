import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import HfArgumentParser

from toxic_x_dom.binary_classification.huggingface import main as binary_classification, ModelArguments, \
    DataTrainingArguments, TrainingArguments
from toxic_x_dom.evaluation import evaluate_token_level

from toxic_x_dom.data import SPAN_DATASETS
from toxic_x_dom.results_db import open_db

load_dotenv()

PROJECT_HOME = os.getenv('TOXIC_X_DOM_HOME')


def perform_attribution(model_args, data_args, training_args):
    """
    Forwards a model on a dataset and does input attribution, returning the dataset extended with attributions.
    """
    out_dir = os.path.join(PROJECT_HOME, 'experiments/binary_classification/outputs/temp/')

    training_args.output_dir = out_dir
    training_args.do_train = False
    training_args.do_eval = False
    training_args.do_predict = True
    training_args.do_attribution = True
    training_args.include_inputs_for_metrics = True
    training_args.predict_split = training_args.attribution_split

    results = binary_classification(model_args, data_args, training_args)
    shutil.rmtree(out_dir)

    attributions = results['attributions']

    print(f"The attribution scores have mean: {attributions[attributions != -100].mean()} "
          f"and standard deviation: {attributions[attributions != -100].std()}")

    attribution_list = []
    for attribution in attributions:
        attribution_list.append(list(attribution[attribution != -100]))

    split = results['attribution_dataset']
    split = split.add_column('attributions', attribution_list)

    if 'predictions' in results:
        predictions_list = list(results['predictions'])
        split = split.add_column('toxic_prediction', predictions_list)

    return split


@dataclass
class RationaleExtractionArguments:

    captum_classes: List[str] = field(
        default_factory=lambda: [
            'captum.attr.DeepLift',
            'captum.attr.IntegratedGradients',
        ],
        metadata={"help": "The list of captum classes to search over."}
    )

    scale_attribution_scores: List[bool] = field(
        default_factory=lambda: [True, False], metadata={"help":  ""}
    )
    cumulative_scoring: List[bool] = field(
        default_factory=lambda: [True, False], metadata={"help": ""}
    )
    propagate_binary_predictions: List[bool] = field(
        default_factory=lambda: [True, False], metadata={"help": ""}
    )

    filling_chars: List[int] = field(
        default_factory=lambda: [0, 1, 9999], metadata={"help": ""}
    )

    min_threshold: float = 0.0
    max_threshold: float = 1.0
    steps_threshold: int = 50


def search_rationale_extraction(attr_args, model_args, data_args, training_args):
    threshold_space = np.linspace(attr_args.min_threshold, attr_args.max_threshold, attr_args.steps_threshold)

    results = []
    for eval_dataset in set(SPAN_DATASETS.keys()):
        data_args.dataset_name = eval_dataset

        # TODO optimize by doing the prediction here instead of repeating for each attribution method

        for captum_class in attr_args.captum_classes:
            training_args.captum_class = captum_class
            method_name = captum_class.split('.')[-1]

            attributed_dataset = perform_attribution(model_args, data_args, training_args)

            for scale_attribution_scores in attr_args.scale_attribution_scores:
                for cumulative_scoring in attr_args.cumulative_scoring:
                    for propagate in attr_args.propagate_binary_predictions:
                        for threshold in threshold_space:
                            for filling_chars in attr_args.filling_chars:

                                if scale_attribution_scores:
                                    pass

                                if cumulative_scoring:
                                    # TODO ...
                                    pass

                                def add_predictions(example):
                                    example['pred_token_toxic_mask'] = [a > threshold for a in example['attributions']]
                                    return example
                                attributed_dataset = attributed_dataset.map(add_predictions)

                                results_dict = evaluate_token_level(
                                    attributed_dataset['pred_token_toxic_mask'], attributed_dataset,
                                    propagate_binary_predictions=propagate,
                                    nr_spaces_to_fill=filling_chars,
                                )

                                results.append(results_dict | {
                                    'eval_dataset': eval_dataset,
                                    'attribution_method': method_name,
                                    'scale_scores': scale_attribution_scores,
                                    'cumulative_scoring': cumulative_scoring,
                                    'propagate_binary': propagate,
                                    'threshold': threshold,
                                    'filling_chars': filling_chars,
                                })

    results_df = pd.DataFrame(results)
    attribution_columns = ['attribution_method', 'scale_scores', 'cumulative_scoring', 'threshold']
    attribution_df = results_df[attribution_columns]
    results_df = results_df.drop(columns=attribution_columns)
    results_df['id'] = [uuid.uuid4() for _ in range(len(results_df.index))]

    train_dataset_col = [model_args.model_name_or_path.split('-')[-1].replace('/', '')] * len(results_df.index)
    results_df['train_dataset'] = train_dataset_col

    db = open_db()
    columns = ','.join(results_df.columns)
    db.execute(f'INSERT INTO evaluation({columns}) SELECT {columns} FROM results_df;')

    attribution_df.insert(0, column='id', value=results_df['id'])
    db.execute('INSERT INTO rationale_evaluation SELECT * FROM attribution_df;')
    db.close()


if __name__ == "__main__":
    parser = HfArgumentParser((RationaleExtractionArguments, ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        _attr_args, _model_args, _data_args, _train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        _attr_args, _model_args, _data_args, _train_args = parser.parse_args_into_dataclasses()

    search_rationale_extraction(_attr_args, _model_args, _data_args, _train_args)
