import argparse

import pandas as pd
from tqdm.auto import tqdm
import numpy as np

from toxic_x_dom.lexicon.lexicon_construction import construct_lexicon, calculate_scores, count_tokens
from toxic_x_dom.data import load_lexicons
from toxic_x_dom.evaluation import evaluate_lexicon

from toxic_x_dom.binary_classification.linear import add_predictions_to_dataset as default_linear
from toxic_x_dom.binary_classification.huggingface import add_predictions_to_dataset as default_huggingface
from toxic_x_dom.results_db import open_db, insert_evaluation, insert_predictions

from toxic_x_dom.data import SPAN_DATASETS

BINARY_TOXICITY_CLASSIFIERS = {
    'count_based_logistic_regression': default_linear,
    'huggingface': default_huggingface,
}


def _generate_grid(existing_lexicons, config):
    MIN_OCCURRENCE = config['min_occurrence_axis']
    FILL_CHARS = config['filling_chars_axis']
    PROP_BINARY = config['prop_binary_axis']
    THETA = np.linspace(config['min_theta'], config['max_theta'], config['steps_theta'])

    trial_configs = []
    for dev_dataset_key in config['datasets']:
        for filling_chars in FILL_CHARS:
            for prop_binary in PROP_BINARY:
                for train_dataset_key in config['datasets']:
                    trial_config_p1 = (dev_dataset_key, filling_chars, prop_binary, train_dataset_key)
                    trial_config_p2 = {}

                    if config['existing_lexicons']:
                        lexicon_keys = list(existing_lexicons.keys())
                        trial_config_p2['existing_lexicons'] = lexicon_keys

                    if config['constructed_lexicons']:
                        constr_trials = []
                        for min_occ in MIN_OCCURRENCE:
                            for theta in THETA:
                                constr_trials.append((min_occ, theta))
                        trial_config_p2['constructed_lexicons'] = constr_trials

                    trial_configs.append((trial_config_p1, trial_config_p2))
    return trial_configs


def eval_trials(trial_configs, config, span_datasets, existing_lexicons, constructed_lexicons):
    split = 'test' if config['eval_on_test'] else 'dev'
    results = []
    predictions = []
    for trial_config in tqdm(trial_configs):
        (eval_dataset_key, filling_chars, prop_binary, train_dataset_key), p2 = trial_config

        dataset = span_datasets[(train_dataset_key, eval_dataset_key)]

        if 'existing_lexicons' in p2:
            for lexicon_key in p2['existing_lexicons']:
                lexicon = existing_lexicons[lexicon_key]
                results_dict = evaluate_lexicon(
                    lexicon, dataset,
                    propagate_binary_predictions=prop_binary,
                    nr_spaces_to_fill=filling_chars,
                    split=split
                )
                results.append(results_dict['metrics'] | {
                    'train_dataset': train_dataset_key,
                    'eval_dataset': f"{eval_dataset_key}-{split}",
                    'min_occurrence': -1,
                    'theta': np.nan,
                    'propagate_binary': prop_binary,
                    'filling_chars': filling_chars,
                    'lexicon_key': lexicon_key,
                })
                predictions.append(results_dict['predictions'])

        if 'constructed_lexicons' in p2:
            for min_occ, theta in p2['constructed_lexicons']:
                lexicon = constructed_lexicons[(train_dataset_key, min_occ, theta)]
                if len(lexicon) > 0:
                    lexicon_tokens, _ = zip(*lexicon)
                else:
                    lexicon_tokens = []
                results_dict = evaluate_lexicon(
                    lexicon_tokens, dataset,
                    propagate_binary_predictions=prop_binary,
                    nr_spaces_to_fill=filling_chars,
                    split=split
                )
                results.append(results_dict['metrics'] | {
                    'train_dataset': train_dataset_key,
                    'eval_dataset': f"{eval_dataset_key}-{split}",
                    'min_occurrence': min_occ,
                    'theta': theta,
                    'propagate_binary': prop_binary,
                    'filling_chars': filling_chars,
                    'lexicon_key': train_dataset_key
                })
                predictions.append(results_dict['predictions'])

    results_df = pd.DataFrame(results)
    results_df = insert_evaluation(results_df)

    insert_predictions(results_df['id'], predictions)

    db = open_db()
    LEXICON_COLUMNS = ['id', 'lexicon_key', 'lexicon_size', 'min_occurrence', 'theta']
    columns = ','.join(LEXICON_COLUMNS)
    db.execute(f'INSERT INTO lexicon_evaluation({columns}) SELECT {columns} FROM results_df;')

    return results, predictions


def construct_lexicons(span_datasets, min_occ_axis, theta_axis):
    scores = {
        (key, min_occ): calculate_scores(*count_tokens(df, minimum_occurrences=min_occ))
        for key, df in {k[0]: df for k, df in span_datasets.items() if k[0] == k[1]}.items()
        for min_occ in min_occ_axis
    }
    constructed_lexicons = {
        (lex_key, min_occ, theta): construct_lexicon(scores[(lex_key, min_occ)], theta=theta)
        for theta in theta_axis
        for min_occ in min_occ_axis
        for lex_key in SPAN_DATASETS.keys()
    }
    return constructed_lexicons


def gridsearch(span_datasets, existing_lexicons, config):
    theta_axis = np.linspace(config['min_theta'], config['max_theta'], config['steps_theta'])
    constructed_lexicons = construct_lexicons(
        span_datasets, config['min_occurrence_axis'], theta_axis
    ) if config['constructed_lexicons'] else None

    grid = _generate_grid(existing_lexicons, config)
    eval_trials(grid, config, span_datasets, existing_lexicons, constructed_lexicons)


def load_datasets(config):
    # train/load binary classifiers
    classifier = BINARY_TOXICITY_CLASSIFIERS[config['binary_toxicity_classifier']]

    _span_datasets = {
        (train_key, eval_key): classifier(eval_key, train_key, split_key='test' if config['eval_on_test'] else 'dev')
        for train_key in SPAN_DATASETS.keys() for eval_key in SPAN_DATASETS.keys()
        if train_key in config['datasets'] and eval_key in config['datasets']
    }
    print(f'F1 scores for binary classifiers on dev split: \n { {key: f1 for key, (_, f1) in _span_datasets.items()} }')

    return {key: data for key, (data, _) in _span_datasets.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # options
    parser.add_argument('--eval_on_test', action='store_true')
    parser.add_argument('--binary_toxicity_classifier',
                        choices=list(BINARY_TOXICITY_CLASSIFIERS.keys()),
                        default='huggingface',
                        type=str)
    parser.add_argument('--skip_existing_lexicons', action='store_false', dest='existing_lexicons')
    parser.add_argument('--skip_constructed_lexicons', action='store_false', dest='constructed_lexicons')
    parser.add_argument('--datasets', default=['hatexplain', 'semeval', 'cad'], nargs="*", type=str)

    # the axes of the grid we search
    parser.add_argument('--min_occurrence_axis', default=[1, 3, 5, 7, 11], nargs='*', type=int)
    parser.add_argument('--filling_chars_axis', default=[0, 1, 9999], nargs='*', type=int)
    parser.add_argument('--prop_binary_axis', choices=[True, False], default=[True, False], nargs='*', type=bool)

    parser.add_argument('--min_theta', default=0.0, type=float)
    parser.add_argument('--max_theta', default=0.95, type=float)
    parser.add_argument('--steps_theta', default=20, type=int)

    args = parser.parse_args()
    _config = {**vars(args)}

    existing = load_lexicons()
    _span_datasets = load_datasets(_config)

    gridsearch(_span_datasets, existing, _config)
