#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import shutil
import random
import sys
import collections
import time
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import pandas as pd
import torch.nn
from datasets import load_dataset, ClassLabel, load_metric

import transformers
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer as HfTrainer,
    TrainingArguments as HfTrainingArguments,
    default_data_collator,
    set_seed, EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint, has_length
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import toxic_x_dom.data

from dotenv import load_dotenv

from toxic_x_dom.rationale_extraction.attribution import Attributer

load_dotenv()

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.24.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

PROJECT_HOME = os.getenv('TOXIC_X_DOM_HOME')


def add_predictions_to_dataset(dataset_name, model_train_dataset, config_key='bert', split_key='dev',
                               return_as_pandas=True):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_path = os.path.join(PROJECT_HOME, f'experiments/binary_classification/outputs/{config_key}-{model_train_dataset}/')
    out_dir = os.path.join(PROJECT_HOME, f'experiments/binary_classification/outputs/temp{int(time.time())}/')
    model_args, data_args, training_args = parser.parse_dict({
        'output_dir':           out_dir,
        'do_train':             False,
        'do_eval':              False,
        'do_predict':           True,
        'predict_split':        split_key,
        'dataset_name':         dataset_name,
        'model_name_or_path':   model_path,
    })
    results = main(model_args, data_args, training_args)
    shutil.rmtree(out_dir)

    if not return_as_pandas:
        return results

    splits = []
    for key, split_data in results['raw_datasets'].items():
        split_data = split_data.to_pandas()
        split_data['split'] = key
        splits.append(split_data)
    dataset = pd.concat(splits)
    dataset.loc[dataset['split'] == split_key, 'prediction'] = results['predictions']

    f1 = results['metrics']['predict_f1']
    return dataset, f1


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    include_nontoxic_samples: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to filter out non-toxic samples from the data."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    predict_split: str = field(
        default='test', metadata={"help": "TODO"}
    )

    attribution_split: str = field(
        default='dev', metadata={"help": "TODO"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class TrainingArguments(HfTrainingArguments):

    do_attribution: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})

    captum_class: str = field(default="captum.attr.LayerActivation", metadata={"help": "Which Captum class to use."})

    balanced_loss_terms: bool = field(
        default=False,
        metadata={
            "help": "Uses weighted loss terms to give each class equal weight in the final loss."
        }
    )
    balanced_data_sampling: bool = field(
        default=False,
        metadata={
            "help": "Used to sample from less frequent classes such that each class has the same prevalence in batches."
        }
    )
    use_early_stopping: bool = field(
        default=True, metadata={"help": "Whether early stopping is used."}
    )
    early_stopping_patience: int = field(
        default=5, metadata={"help": "For how many epochs in a row the performance must worsen "
                                     "for early stopping to be triggered."}
    )
    early_stopping_threshold: float = field(
        default=0.0, metadata={"help": "TODO"}
    )


class BalancedTrainer(HfTrainer):
    args: TrainingArguments

    def __init__(self, train_class_counts=None, **kwargs):
        super(BalancedTrainer, self).__init__(**kwargs)

        if self.args.balanced_loss_terms or self.args.balanced_data_sampling:
            if train_class_counts is None:
                raise ValueError('Need the class counts to do balancing of data sampling or loss terms.')

            self.train_class_counts = train_class_counts
            print("Class counts: ", train_class_counts)

            total_count = sum(train_class_counts.values())
            normalized_dist = [count / total_count for c, count in sorted(train_class_counts.items(), key=lambda x: x[0])]
            self.class_weights = [1 / (p * len(normalized_dist)) for p in normalized_dist]
            print("Class weights:", self.class_weights)

        if self.args.balanced_loss_terms:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, device=self.model.device))

    def compute_loss(self, model, inputs, return_outputs=False):
        if 'labels' in inputs:
            labels = inputs.get('labels')

        outputs = model(**inputs)
        if self.args.balanced_loss_terms:
            logits = outputs.get('logits')
            loss = self.loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.get('loss')
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if not self.args.balanced_data_sampling:
            return super(BalancedTrainer, self)._get_train_sampler()

        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        if self.args.world_size <= 1:
            weights = [self.class_weights[int(label)] for label in self.train_dataset['toxic']]
            length = 2 * min(self.train_class_counts.values())
            return WeightedRandomSampler(weights, length, replacement=False, generator=generator)
        else:
            raise NotImplementedError()


def main(model_args, data_args, training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.balanced_loss_terms and training_args.balanced_data_sampling:
        raise ValueError("Don't use loss-term balancing and balanced data-sampling together.")

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    raw_datasets = load_dataset(
        toxic_x_dom.data.__file__,
        dataset_name=data_args.dataset_name
    )

    if not data_args.include_nontoxic_samples:
        raw_datasets = raw_datasets.filter(lambda sample: sample['toxic'])
        if training_args.do_train:
            raise ValueError("Don't train on data with only one class...")

    class_label = ClassLabel(names=['non-toxic', 'toxic'])
    label_list = class_label.names

    # Labels
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples["full_text"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )
        tokenized['char_offsets'] = [encoding.offsets for encoding in tokenized.encodings]
        tokenized['label'] = list(map(
            lambda sample_toxic:
            class_label.str2int('toxic') if sample_toxic else class_label.str2int('non-toxic'), examples['toxic']
        ))
        return tokenized

    train_class_counts = None

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            train_class_counts = collections.Counter(train_dataset['label'])

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if "dev" not in raw_datasets:
            raise ValueError("--do_eval requires a development dataset")
        eval_dataset = raw_datasets["dev"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="development dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on development dataset",
            )

    if training_args.do_predict:
        if data_args.predict_split not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets[data_args.predict_split]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    if training_args.do_attribution:
        if data_args.attribution_split not in raw_datasets:
            raise ValueError("TODO")
        attribution_dataset = raw_datasets[training_args.attribution_split]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(attribution_dataset), data_args.max_predict_samples)
            attribution_dataset = attribution_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            attribution_dataset = attribution_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on attribution dataset",
            )
            nr_max_length = sum(1 for x in attribution_dataset['input_ids']
                                if len(x) == model.config.max_position_embeddings)
            logger.warning(
                f"Number of max-length samples: {nr_max_length} out of {len(attribution_dataset)}. "
                f"These won't get attribution to characters that are truncated, leading to suboptimal performance."
            )

    # Get the metric function
    metric = load_metric("f1")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    callbacks = []
    if training_args.use_early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
            early_stopping_threshold=training_args.early_stopping_threshold,
        ))

    # Initialize our Trainer
    trainer = BalancedTrainer(
        train_class_counts=train_class_counts,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )

    return_values = {'raw_datasets': raw_datasets}

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")

        return_values['predictions'] = predictions
        return_values['metrics'] = metrics

    # Attribution
    if training_args.do_attribution:
        logger.info('*** Input Attribution ***')
        return_values['attribution_dataset'] = attribution_dataset

        attributer = Attributer(None, training_args, model, tokenizer, data_collator)
        results = attributer.attribute(attribution_dataset)
        return_values['attributions'] = results.predictions

    return return_values


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        _model_args, _data_args, _training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        _model_args, _data_args, _training_args = parser.parse_args_into_dataclasses()

    main(_model_args, _data_args, _training_args)
