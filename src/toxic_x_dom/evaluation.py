import itertools
import re
from typing import Set

import numpy as np

from tqdm import tqdm

from toxic_x_dom.utils import list_of_lists_to_numpy


def metrics(pred: Set, label: Set):
    correct = pred & label
    empty_label = empty_pred = empty_both = 0

    if len(pred) == 0 and len(label) > 0:
        # only prediction is empty
        p = float('nan')        # precision undefined
        r = 0                   # = len(correct) / len(label)
        f1 = 0                  # prediction was empty, label was not, so count as 0
        empty_pred = 1
    elif len(label) == 0 and len(pred) > 0:
        # only label is empty
        p = 0                   # = len(correct) / len(pred)
        r = float('nan')        # recall undefined (nothing to recall), not counted in average
        f1 = 0                  # label was empty, prediction was not, so count as 0
        empty_label = 1
    elif len(label) == 0 and len(pred) == 0:
        # both are empty
        p = r = float('nan')    # precision and recall are undefined
        f1 = 1                  # correctly predicted no toxic language, so count as 1
        empty_both = 1
    elif len(correct) == 0:
        # only intersection is empty
        p = r = f1 = 0          # complete mismatch
    else:
        p = len(correct) / len(pred)
        r = len(correct) / len(label)
        f1 = 2 * p * r / (p + r)

    return f1, p, r, (empty_pred, empty_label, empty_both)


def evaluate_token_level(token_prediction_mask, dataset):
    token_char_offsets = dataset['char_offsets']
    char_label_mask = dataset['toxic_mask']
    toxic_mask = np.array(dataset['toxic'])
    span_mask = list_of_lists_to_numpy(dataset['toxic_mask']).astype(np.bool).any(axis=-1)

    nr_rows = len(token_prediction_mask)
    char_idxs = np.arange(max(max(o[1] for o in offsets if o is not None) for offsets in token_char_offsets))
    f1, p, r = np.full(nr_rows, np.nan), np.full(nr_rows, np.nan), np.full(nr_rows, np.nan)
    for i in range(nr_rows):
        offsets = token_char_offsets[i]

        def convert_mask_to_char_level(mask):
            # use token's character offsets to convert predictions
            return np.array(list(itertools.chain.from_iterable([
                [t_toxic] * (os0[1] - os0[0])   # repeat prediction assigned to token
                + [False] * (os1[0] - os0[1])   # interject False for characters that aren't part of any token
                for t_toxic, os0, os1 in zip(mask, offsets, offsets[1:]+[(offsets[-1][0],)])
            ])))

        char_prediction_mask = convert_mask_to_char_level(token_prediction_mask[i])

        l, = char_prediction_mask.shape
        if l > 0:
            pred_chars_offsets = set(char_idxs[:l][char_prediction_mask])
            label_chars_offsets = set(char_idxs[:l][char_label_mask[i][:l]])
        else:
            pred_chars_offsets = label_chars_offsets = set()
        _f1, _p, _r, _ = metrics(pred_chars_offsets, label_chars_offsets)
        f1[i] = _f1
        p[i] = _p
        r[i] = _r
    return {
        'F1 (micro)': np.nanmean(f1),
        'Precision (micro)': np.nanmean(p),
        'Recall (micro)': np.nanmean(r),
        'F1 (toxic)': np.nanmean(f1[toxic_mask]),
        'Precision (toxic)': np.nanmean(p[toxic_mask]),
        'Recall (toxic)': np.nanmean(r[toxic_mask]),
        'F1 (toxic-no_span)': np.nanmean(f1[toxic_mask & ~span_mask]),
        'Precision (toxic-no_span)': np.nanmean(p[toxic_mask & ~span_mask]),
        'Recall (toxic-no_span)': np.nanmean(r[toxic_mask & ~span_mask]),
        'F1 (non-toxic)': np.nanmean(f1[~toxic_mask]),
        'Precision (non-toxic)': np.nanmean(p[~toxic_mask]),
        'Recall (non-toxic)': np.nanmean(r[~toxic_mask]),
    }


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

        f1, p, r, (empty_pred, empty_label, empty_both) = metrics(pred, label)
        nr_empty_pred += empty_pred
        nr_empty_label += empty_label
        nr_empty_both += empty_both

        split.at[index, 'Precision'] = p
        split.at[index, 'Recall'] = r
        split.at[index, 'F1'] = f1
        split.at[index, '% Predicted'] = len(pred) / len(full_text) if len(full_text) > 0 else float('nan')

    results = {
        'F1 (micro)': split['F1'].mean(),
        'Precision (micro)': split['Precision'].mean(),
        'Recall (micro)': split['Recall'].mean(),
        'F1 (toxic)': split.loc[split['toxic'], 'F1'].mean(),
        'Precision (toxic)': split.loc[split['toxic'], 'Precision'].mean(),
        'Recall (toxic)': split.loc[split['toxic'], 'Recall'].mean(),
        'F1 (toxic-no_span)': split.loc[split['toxic'] & ~split['toxic_mask'].any(), 'F1'].mean(),
        'Precision (toxic-no_span)': split.loc[split['toxic'] & ~split['toxic_mask'].any(), 'Precision'].mean(),
        'Recall (toxic-no_span)': split.loc[split['toxic'] & ~split['toxic_mask'].any(), 'Recall'].mean(),
        'F1 (non-toxic)': split.loc[~split['toxic'], 'F1'].mean(),
        'Precision (non-toxic)': split.loc[~split['toxic'], 'Precision'].mean(),
        'Recall (non-toxic)': split.loc[~split['toxic'], 'Recall'].mean(),
        'non-toxic accuracy': nr_empty_both / (nr_empty_label + nr_empty_both),
        'non-toxic %-predicted': split.loc[~split['toxic'], '% Predicted'].mean(),
        'nr_empty_pred': nr_empty_pred,
        'nr_empty_label': nr_empty_label,
        'nr_empty_both': nr_empty_both,
        'nr_samples': len(split),
        'lexicon size': len(lexicon_tokens),
    }

    return results
