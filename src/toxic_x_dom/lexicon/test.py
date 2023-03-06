from toxic_x_dom.data import load_lexicons, SPAN_DATASETS
from toxic_x_dom.lexicon.search import load_datasets, construct_lexicons, eval_trials
from toxic_x_dom.results_db import open_db


def eval_on_test():
    db = open_db()

    # get best performing trials
    columns = ",".join(['eval_dataset', 'train_dataset', 'propagate_binary', 'filling_chars',
                       'lexicon_key', 'min_occurrence', 'theta'])
    results = db.query(f"SELECT {columns} FROM trial_evaluations_to_test "
                       f" WHERE method_type = 'lexicon'").df()

    print('Running eval on test split for best trials:')
    print(results)

    lex_config = {
        'binary_toxicity_classifier': 'huggingface',
        'datasets': SPAN_DATASETS.keys(),
        'eval_on_test': True,
    }

    # load/calc requisite data
    span_datasets = load_datasets(lex_config)
    constructed = construct_lexicons(span_datasets, results['min_occurrence'].unique(), results['theta'].unique())
    existing = load_lexicons()

    # construct test configs
    configs = []
    for _, row in results.iterrows():
        config_p1 = (row.eval_dataset, row.filling_chars, row.propagate_binary, row.train_dataset)

        if row.lexicon_key in existing.keys():
            config_p2 = {'existing_lexicons': [row.lexicon_key]}
        else:
            config_p2 = {'constructed_lexicons': [(row.min_occurrence, row.theta)]}

        configs.append((config_p1, config_p2))

    if len(configs) > 0:
        eval_trials(configs, lex_config, span_datasets, existing, constructed)


if __name__ == "__main__":
    eval_on_test()
