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


def gridsearch(span_datasets, existing_lexicons, config):
    MIN_OCCURRENCE = config['min_occurrence_axis']
    FILL_CHARS = config['filling_chars_axis']
    PROP_BINARY = config['prop_binary_axis']

    THETA = np.linspace(config['min_theta'], config['max_theta'], config['steps_theta'])

    if config['constructed_lexicons']:
        scores = {
            (key, min_occ): calculate_scores(*count_tokens(df, minimum_occurrences=min_occ))
            for key, df in {k[0]: df for k, df in span_datasets.items() if k[0] == k[1]}.items()
            for min_occ in MIN_OCCURRENCE
        }
        constructed_lexicons = {
            (lex_key, min_occ, theta): construct_lexicon(scores[(lex_key, min_occ)], theta=theta)
            for theta in THETA
            for min_occ in MIN_OCCURRENCE
            for lex_key in SPAN_DATASETS.keys()
        }

    results = []
    predictions = []

    total_steps = len(SPAN_DATASETS) * len(FILL_CHARS) * len(PROP_BINARY) * (
            (len(SPAN_DATASETS) * len(MIN_OCCURRENCE) * config['steps_theta'] if config['constructed_lexicons'] else 0)
            + (len(existing_lexicons) if config['existing_lexicons'] else 0)
    )
    pbar_total = tqdm(total=total_steps, desc='Overall Progress')

    # for evaluation datasets
    pbar1 = tqdm(SPAN_DATASETS.keys(), desc='Evaluation dataset', leave=False)
    for dev_dataset_key in pbar1:
        pbar1.set_postfix({'key': dev_dataset_key})

        pbar2 = tqdm(FILL_CHARS, desc='filling_chars', leave=False)
        for filling_chars in pbar2:
            pbar2.set_postfix({'?': str(filling_chars)})

            pbar5 = tqdm(PROP_BINARY, desc='Propagate predictions from binary model?', leave=False)
            for prop_binary in pbar5:
                pbar5.set_postfix({'prop?': str(prop_binary)})

                pbar0 = tqdm(SPAN_DATASETS.keys(), desc="'Train' Dataset",  leave=False)
                for train_dataset_key in pbar0:
                    pbar0.set_postfix({'key': train_dataset_key})

                    dataset = span_datasets[(train_dataset_key, dev_dataset_key)]

                    if config['existing_lexicons']:
                        # for existing lexicons
                        for lexicon_key, lexicon in existing_lexicons.items():
                            results_dict = evaluate_lexicon(
                                lexicon, dataset,
                                propagate_binary_predictions=prop_binary,
                                nr_spaces_to_fill=filling_chars
                            )
                            results.append(results_dict['metrics'] | {
                                'train_dataset': train_dataset_key,
                                'eval_dataset': dev_dataset_key,
                                'min_occurrence': -1,
                                'theta': np.nan,
                                'propagate_binary': prop_binary,
                                'filling_chars': filling_chars,
                                'lexicon_key': lexicon_key,
                            })
                            predictions.append(results_dict['predictions'])
                            pbar_total.update()

                    if config['constructed_lexicons']:
                        pbar3 = tqdm(MIN_OCCURRENCE, desc='Min occurrence', leave=False)
                        for min_occ in pbar3:
                            pbar3.set_postfix({'min_occ': min_occ})

                            pbar4 = tqdm(THETA, total=config['steps_theta'], desc='Theta',leave=False)
                            for theta in pbar4:
                                pbar4.set_postfix({'θ': theta})
                                lexicon = constructed_lexicons[(train_dataset_key, min_occ, theta)]
                                if len(lexicon) > 0:
                                    lexicon_tokens, _ = zip(*lexicon)
                                else:
                                    lexicon_tokens = []
                                results_dict = evaluate_lexicon(
                                    lexicon_tokens, dataset,
                                    propagate_binary_predictions=prop_binary,
                                    nr_spaces_to_fill=filling_chars
                                )
                                results.append(results_dict['metrics'] | {
                                    'train_dataset': train_dataset_key,
                                    'eval_dataset': dev_dataset_key,
                                    'min_occurrence': min_occ,
                                    'theta': theta,
                                    'propagate_binary': prop_binary,
                                    'filling_chars': filling_chars,
                                    'lexicon_key': train_dataset_key
                                })
                                predictions.append(results_dict['predictions'])
                                pbar_total.update()

    results_df = pd.DataFrame(results)
    results_df = insert_evaluation(results_df)

    insert_predictions(results_df['id'], predictions)

    db = open_db()

    LEXICON_COLUMNS = ['id', 'lexicon_key', 'lexicon_size', 'min_occurrence', 'theta']
    columns = ','.join(LEXICON_COLUMNS)
    db.execute(f'INSERT INTO lexicon_evaluation({columns}) SELECT {columns} FROM results_df;')

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # options
    parser.add_argument('--results_file', default='lexicon_results.csv', type=str)
    parser.add_argument('--binary_toxicity_classifier',
                        choices=list(BINARY_TOXICITY_CLASSIFIERS.keys()),
                        default='huggingface',
                        type=str)
    parser.add_argument('--skip_existing_lexicons', action='store_false', dest='existing_lexicons')
    parser.add_argument('--skip_constructed_lexicons', action='store_false', dest='constructed_lexicons')

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

    # train/load binary classifiers
    classifier = BINARY_TOXICITY_CLASSIFIERS[_config['binary_toxicity_classifier']]
    _span_datasets = {
        (train_key, eval_key): classifier(eval_key, train_key)
        for train_key in SPAN_DATASETS.keys() for eval_key in SPAN_DATASETS.keys()
    }

    print(f'F1 scores for binary classifiers on dev split: \n { {key: f1 for key, (_, f1) in _span_datasets.items()} }')

    gridsearch(
        {key: func for key, (func, _) in _span_datasets.items()},
        existing, _config
    )
