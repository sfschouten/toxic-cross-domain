import re
import numpy as np

from tqdm import tqdm


def evaluate_lexicon(lexicon_tokens, df, split='dev', join_predicted_words=True, propagate_binary_predictions=True):
    split = df[df['split'] == split].copy()
    split['Precision'] = np.nan
    split['Recall'] = np.nan
    split['F1'] = np.nan
    split['% Predicted'] = np.nan

    nr_empty_pred = nr_empty_label = nr_empty_both = 0
    for index, row in tqdm(split.iterrows(), total=len(split), leave=False):
        full_text = row.full_text

        label = set(i for i, b in enumerate(row.toxic_mask) if b)
        pred = set()

        if not propagate_binary_predictions or row.prediction:
            # The binary model predicted toxic, use lexicon for span
            expr = "|".join([re.escape(token) for token in lexicon_tokens])
            for match in re.finditer(expr, full_text, re.I):
                pred.update(set(range(match.start(), match.end())))

            if join_predicted_words and len(pred) > 0:
                min_ = min(pred)
                max_ = max(pred)
                pred = set(range(min_, max_ + 1))

        correct = label & pred

        if len(correct) > 0:
            p = len(correct) / len(pred)
            r = len(correct) / len(label)
            f1 = 2 * p * r / (p + r)
        # len(correct) == 0
        elif len(label) > 0:  # prediction empty
            p = float('nan')  # precision undefined
            r = 0  # = len(correct) / len(label)
            f1 = 0  # prediction was empty, label was not, so count as 0
            nr_empty_pred += 1
        elif len(pred) > 0:  # label empty
            p = 0  # = len(correct) / len(pred)
            r = float('nan')  # recall undefined, not counted in average
            f1 = 0  # label was empty, prediction was not, so count as 0
            nr_empty_label += 1
        else:  # prediction and label are empty
            p = r = f1 = 1  # correctly predicted no toxic language, so count as 1
            nr_empty_both += 1

        split.at[index, 'Precision'] = p
        split.at[index, 'Recall'] = r
        split.at[index, 'F1'] = f1
        split.at[index, '% Predicted'] = len(pred) / len(full_text) if len(full_text) > 0 else float('nan')

    results = {
        'Precision (toxic)': split.loc[split['toxic'], 'Precision'].mean(),
        'Precision (non-toxic)': split.loc[~split['toxic'], 'Precision'].mean(),
        'Precision (micro)': split['Precision'].mean(),
        'Recall (toxic)': split.loc[split['toxic'], 'Recall'].mean(),
        'Recall (non-toxic)': split.loc[~split['toxic'], 'Recall'].mean(),
        'Recall (micro)': split['Recall'].mean(),
        'F1 (toxic)': split.loc[split['toxic'], 'F1'].mean(),
        'F1 (non-toxic)': split.loc[~split['toxic'], 'F1'].mean(),
        'F1 (micro)': split['F1'].mean(),
        'non-toxic accuracy': nr_empty_both / (nr_empty_label + nr_empty_both),
        'non-toxic %-predicted': split.loc[~split['toxic'], '% Predicted'].mean(),
        'nr_empty_pred': nr_empty_pred,
        'nr_empty_label': nr_empty_label,
        'nr_empty_both': nr_empty_both,
        'nr_samples': len(split),
        'lexicon size': len(lexicon_tokens),
    }

    return results
