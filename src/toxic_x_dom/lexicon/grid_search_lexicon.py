import argparse
import csv
import uuid

import pandas as pd
from tqdm.auto import tqdm
import numpy as np

from toxic_x_dom.lexicon.lexicon_construction import construct_lexicon, calculate_scores, count_tokens
from toxic_x_dom.data import load_lexicons
from toxic_x_dom.evaluation import evaluate_lexicon

from toxic_x_dom.binary_classification.linear import add_predictions_to_dataset as default_linear
from toxic_x_dom.binary_classification.huggingface import add_predictions_to_dataset as default_huggingface
from toxic_x_dom.results_db import open_db

BINARY_TOXICITY_CLASSIFIERS = {
    'count_based_logistic_regression': default_linear,
    'huggingface': default_huggingface,
}


def gridsearch(span_datasets, existing_lexicons, config):
    MIN_OCCURRENCE = config['min_occurrence_axis']
    JOIN_PREDICTED = config['join_predicted_axis']
    PROP_BINARY = config['join_predicted_axis']

    THETA = np.linspace(config['min_theta'], config['max_theta'], config['steps_theta'])
    RESULTS_COLUMNS = [
        'F1 (micro)', 'Precision (micro)', 'Recall (micro)', 'F1 (toxic)', 'Precision (toxic)', 'Recall (toxic)',
        'F1 (toxic-no_span)', 'Precision (toxic-no_span)', 'Recall (toxic-no_span)', 'F1 (non-toxic)',
        'Precision (non-toxic)', 'Recall (non-toxic)', 'lexicon size']
    #'non-toxic accuracy', 'non-toxic %-predicted', 'nr_empty_pred',
    #    'nr_empty_label', 'nr_empty_both', 'nr_samples',
    #]

    if config['constructed_lexicons']:
        scores = {
            (key, min_occ): calculate_scores(*count_tokens(df, minimum_occurrences=min_occ))
            for key, df in span_datasets.items()
            for min_occ in MIN_OCCURRENCE
        }
        constructed_lexicons = {
            (lex_key, min_occ, theta): construct_lexicon(scores[(lex_key, min_occ)], theta=theta)
            for theta in THETA
            for min_occ in MIN_OCCURRENCE
            for lex_key in span_datasets.keys()
        }

    results = []

    total_steps = len(span_datasets) * len(JOIN_PREDICTED) * len(PROP_BINARY) * (
            (len(span_datasets) * len(MIN_OCCURRENCE) * config['steps_theta'] if config['constructed_lexicons'] else 0)
            + (len(existing_lexicons) if config['existing_lexicons'] else 0)
    )
    pbar_total = tqdm(total=total_steps, desc='Overall Progress')

    # for evaluation datasets
    pbar1 = tqdm(span_datasets.items(), desc='Evaluation dataset', leave=False)
    for dev_dataset_key, dev_df in pbar1:
        pbar1.set_postfix({'key': dev_dataset_key})

        pbar2 = tqdm(JOIN_PREDICTED, desc='Join predicted words?', leave=False)
        for join_predicted in pbar2:
            pbar2.set_postfix({'?': str(join_predicted)})

            pbar5 = tqdm(PROP_BINARY, desc='Propagate predictions from binary model?', leave=False)
            for prop_binary in pbar5:
                pbar5.set_postfix({'prop?': str(prop_binary)})

                if config['existing_lexicons']:
                    # for existing lexicons
                    for lexicon_key, lexicon in existing_lexicons.items():
                        results_dict = evaluate_lexicon(
                            lexicon, dev_df,
                            join_predicted_words=join_predicted,
                            propagate_binary_predictions=prop_binary,
                        )
                        results.append(
                            [lexicon_key, dev_dataset_key, join_predicted, -1, np.nan, prop_binary]
                            + [results_dict[key] for key in RESULTS_COLUMNS]
                        )
                        pbar_total.update()

                if config['constructed_lexicons']:
                    # for constructed lexicons
                    pbar0 = tqdm(span_datasets.items(), desc="'Train' Dataset",  leave=False)
                    for lexicon_key, _ in pbar0:
                        pbar0.set_postfix({'key': lexicon_key})

                        pbar3 = tqdm(MIN_OCCURRENCE, desc='Min occurrence', leave=False)
                        for min_occ in pbar3:
                            pbar3.set_postfix({'min_occ': min_occ})

                            pbar4 = tqdm(THETA, total=config['steps_theta'], desc='Theta',leave=False)
                            for theta in pbar4:
                                pbar4.set_postfix({'θ': theta})
                                lexicon = constructed_lexicons[(lexicon_key, min_occ, theta)]
                                if len(lexicon) > 0:
                                    lexicon_tokens, _ = zip(*lexicon)
                                else:
                                    lexicon_tokens = []
                                results_dict = evaluate_lexicon(
                                    lexicon_tokens, dev_df,
                                    join_predicted_words=join_predicted,
                                    propagate_binary_predictions=prop_binary,
                                )

                                results.append(
                                    [lexicon_key, dev_dataset_key, join_predicted, min_occ, theta, prop_binary]
                                    + [results_dict[key] for key in RESULTS_COLUMNS])
                                pbar_total.update()

    results_df = pd.DataFrame(
        results,
        columns=['train_dataset', 'eval_dataset', 'join_predicted', 'min_occurrence', 'theta', 'propagate_binary',
                 'f1_micro', 'precision_micro', 'recall_micro', 'f1_toxic', 'precision_toxic',
                 'recall_toxic', 'f1_toxic_no_span', 'precision_toxic_no_span', 'recall_toxic_no_span', 'f1_non_toxic',
                 'precision_non_toxic', 'recall_non_toxic', 'lexicon_size']
    )
    lexicon_columns = ['lexicon_size', 'min_occurrence', 'theta']
    lexicon_df = results_df[lexicon_columns]
    results_df = results_df.drop(columns=lexicon_columns+['join_predicted'])
    results_df['id'] = [uuid.uuid4() for _ in range(len(results_df.index))]
    results_df['filling_chars'] = 1001          # TODO ...

    db = open_db()
    columns = ','.join(results_df.columns)
    db.execute(f'INSERT INTO evaluation({columns}) SELECT {columns} FROM results_df;')

    lexicon_df.insert(0, column='id', value=results_df['id'])
    db.execute('INSERT INTO lexicon_evaluation SELECT * FROM lexicon_df;')
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # options
    parser.add_argument('--results_file', default='lexicon_results.csv', type=str)
    parser.add_argument('--binary_toxicity_classifier',
                        choices=list(BINARY_TOXICITY_CLASSIFIERS.keys()),
                        default='count_based_logistic_regression',
                        type=str)
    parser.add_argument('--skip_existing_lexicons', action='store_false', dest='existing_lexicons')
    parser.add_argument('--skip_constructed_lexicons', action='store_false', dest='constructed_lexicons')

    # the axes of the grid we search
    parser.add_argument('--min_occurrence_axis', default=[1, 3, 5, 7, 11], nargs='*', type=int)
    parser.add_argument('--join_predicted_axis', choices=[True, False], default=[True, False], nargs='*', type=bool)
    parser.add_argument('--prop_binary_axis', choices=[True, False], default=[True, False], nargs='*', type=bool)

    parser.add_argument('--min_theta', default=0.0, type=float)
    parser.add_argument('--max_theta', default=0.95, type=float)
    parser.add_argument('--steps_theta', default=21, type=int)

    args = parser.parse_args()
    _config = {**vars(args)}

    existing = load_lexicons()

    from toxic_x_dom.data import SPAN_DATASETS

    # train/load binary classifiers
    _span_datasets = {
        key: BINARY_TOXICITY_CLASSIFIERS[_config['binary_toxicity_classifier']](key) for key in SPAN_DATASETS.keys()
    }

    print(f'F1 scores for binary classifiers on dev split: \n { {key: f1 for key, (_, f1) in _span_datasets.items()} }')

    gridsearch(
        {key: func for key, (func, _) in _span_datasets.items()},
        existing, _config
    )
