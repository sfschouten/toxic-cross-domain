import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
from dotenv import load_dotenv
from transformers import HfArgumentParser

from toxic_x_dom.binary_classification.huggingface import main as binary_classification, ModelArguments, \
    DataTrainingArguments, TrainingArguments
from toxic_x_dom.evaluation import evaluate_token_level

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
    training_args.do_predict = False
    training_args.do_attribution = True
    training_args.include_inputs_for_metrics = True

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
        default_factory=lambda: [True, False],
        metadata={"help":  ""}
    )
    cumulative_scoring: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": ""}
    )

    min_threshold: float = 0.0
    max_threshold: float = 1.0
    steps_threshold: int = 50


def search_rationale_extraction(attr_args, model_args, data_args, training_args):
    threshold_space = np.linspace(attr_args.min_threshold, attr_args.max_threshold, attr_args.steps_threshold)

    for captum_class in attr_args.captum_classes:

        training_args.captum_class = captum_class
        attributed_dataset = perform_attribution(model_args, data_args, training_args)

        for scale_attribution_scores in attr_args.scale_attribution_scores:
            for cumulative_scoring in attr_args.cumulative_scoring:
                for threshold in threshold_space:

                    if scale_attribution_scores:
                        pass

                    if cumulative_scoring:
                        pass

                    def add_predictions(example):
                        example['pred_token_toxic_mask'] = [a > threshold for a in example['attributions']]
                        return example
                    attributed_dataset = attributed_dataset.map(add_predictions)

                    results = evaluate_token_level(
                        attributed_dataset['pred_token_toxic_mask'],
                        attributed_dataset,
                    )
                    print(results)
                    pass


if __name__ == "__main__":
    parser = HfArgumentParser((RationaleExtractionArguments, ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        _attr_args, _model_args, _data_args, _train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        _attr_args, _model_args, _data_args, _train_args = parser.parse_args_into_dataclasses()

    search_rationale_extraction(_attr_args, _model_args, _data_args, _train_args)
