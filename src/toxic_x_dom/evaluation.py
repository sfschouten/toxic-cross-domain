import itertools
import re
from enum import IntFlag
from typing import Set

import numpy as np
import pandas as pd

from tqdm import tqdm

from toxic_x_dom.utils import list_of_lists_to_numpy


class MetricEmptySetResult(IntFlag):
    PRED = 2
    LABEL = 1
    NONE = 0
    BOTH = PRED | LABEL


def metrics(pred: Set, label: Set):
    correct = pred & label
    empty_result = MetricEmptySetResult.NONE

    if len(pred) == 0 and len(label) > 0:
        # only prediction is empty
        p = float('nan')        # precision undefined
        r = 0                   # = len(correct) / len(label)
        f1 = 0                  # prediction was empty, label was not, so count as 0
        empty_result = MetricEmptySetResult.PRED
    elif len(label) == 0 and len(pred) > 0:
        # only label is empty
        p = 0                   # = len(correct) / len(pred)
        r = float('nan')        # recall undefined (nothing to recall), not counted in average
        f1 = 0                  # label was empty, prediction was not, so count as 0
        empty_result = MetricEmptySetResult.LABEL
    elif len(label) == 0 and len(pred) == 0:
        # both are empty
        p = r = float('nan')    # precision and recall are undefined
        f1 = 1                  # correctly predicted no toxic language, so count as 1
        empty_result = MetricEmptySetResult.BOTH
    elif len(correct) == 0:
        # only intersection is empty
        p = r = f1 = 0          # complete mismatch
    else:
        p = len(correct) / len(pred)
        r = len(correct) / len(label)
        f1 = 2 * p * r / (p + r)

    return f1, p, r, empty_result


def _fill_empty_spaces(prediction, max_nr_characters=1):
    sorted_ = sorted(prediction)
    to_add = [set(range(c1+1, c2)) for c1, c2 in zip(sorted_[:-1], sorted_[1:]) if c2 - (c1+1) <= max_nr_characters]
    prediction |= set(itertools.chain.from_iterable(to_add))
    return prediction


def _calc_aggregate_metrics(f1, p, r, toxic_mask, span_mask, empty_pred, empty_label, pct_predicted):
    """

    """
    return {
        'f1_micro':                 np.nanmean(f1),
        'precision_micro':          np.nanmean(p),
        'recall_micro':             np.nanmean(r),
        'f1_toxic':                 np.nanmean(f1[toxic_mask]),
        'precision_toxic':          np.nanmean(p[toxic_mask]),
        'recall_toxic':             np.nanmean(r[toxic_mask]),
        'f1_toxic_no_span':         np.nanmean(f1[toxic_mask & ~span_mask]),
        'precision_toxic_no_span':  np.nanmean(p[toxic_mask & ~span_mask]),
        'recall_toxic_no_span':     np.nanmean(r[toxic_mask & ~span_mask]),
        'f1_non_toxic':             np.nanmean(f1[~toxic_mask]),
        'precision_non_toxic':      np.nanmean(p[~toxic_mask]),
        'recall_non_toxic':         np.nanmean(r[~toxic_mask]),
        'non_toxic_pct_predicted':  np.nanmean(pct_predicted[~toxic_mask]),
        'nr_empty_pred':            np.nansum(empty_pred),
        'nr_empty_label':           np.nansum(empty_label),
        'nr_empty_both':            np.nansum(empty_pred & empty_label),
        'nr_samples':               len(f1),
    }


def evaluate_token_level(
        token_prediction_mask, dataset,
        propagate_binary_predictions=False, nr_spaces_to_fill=1,
):
    token_char_offsets = dataset['char_offsets']
    char_label_mask = dataset['toxic_mask']
    toxic_mask = np.array(dataset['toxic'])
    span_mask = list_of_lists_to_numpy(dataset['toxic_mask']).astype(np.bool).any(axis=-1)
    char_prediction_masks = []

    has_toxic_prediction = 'toxic_prediction' in dataset.features
    if propagate_binary_predictions and not has_toxic_prediction:
        raise ValueError("Toxicity predictions are required if they are to be propagated.")

    char_idxs = np.arange(max(len(ch_mask) for ch_mask in char_label_mask))
    nr_rows = len(token_prediction_mask)
    pct_predicted = np.full(nr_rows, np.nan)
    f1, p, r = np.full(nr_rows, np.nan), np.full(nr_rows, np.nan), np.full(nr_rows, np.nan)
    empty_pred, empty_label = np.full(nr_rows, False), np.full(nr_rows, False)
    for i in range(nr_rows):
        l = len(char_label_mask[i])
        label_chars_offsets = set(char_idxs[:l][char_label_mask[i]])
        pred_chars_offsets = set()
        char_prediction_mask = [False] * l

        if (not propagate_binary_predictions or dataset['toxic_prediction'][i]) and l > 0:
            offsets = token_char_offsets[i]

            def convert_mask_to_char_level(mask):
                # use token's character offsets to convert predictions
                m_offsets = [
                    (m, os0, os1) for (m, os0, os1) in zip(mask, offsets, offsets[1:]+[(offsets[-1][0],)])
                    if os0 != [0, 0]            # remove special tokens and their offsets
                ]
                result = list(itertools.chain.from_iterable([
                    [t_toxic] * (os0[1] - os0[0])   # repeat prediction assigned to token
                    + [False] * (os1[0] - os0[1])   # interject False for characters that aren't part of any token
                    for t_toxic, os0, os1 in m_offsets
                ]))
                result = [False] * m_offsets[0][1][0] + result         # prepend False in case 1st token is not 1st char
                result = result + [False] * (l - m_offsets[-1][1][1])  # append False in case last token is not last char
                return result

            char_prediction_mask = convert_mask_to_char_level(token_prediction_mask[i])
            pred_chars_offsets = set(char_idxs[:l][np.array(char_prediction_mask)])
            pred_chars_offsets = _fill_empty_spaces(pred_chars_offsets, nr_spaces_to_fill)

        char_prediction_masks.append(char_prediction_mask)

        f1[i], p[i], r[i], empty_result = metrics(pred_chars_offsets, label_chars_offsets)
        empty_pred[i] = MetricEmptySetResult.PRED in empty_result
        empty_label[i] = MetricEmptySetResult.LABEL in empty_result
        pct_predicted[i] = len(pred_chars_offsets) / l if l > 0 else float('nan')

    aggregate_metrics = _calc_aggregate_metrics(f1, p, r, toxic_mask, span_mask, empty_pred, empty_label, pct_predicted)
    predictions = pd.DataFrame({
        'sample_id': dataset['id'],
        'span_prediction': char_prediction_masks,
        'toxic_prediction': dataset['toxic_prediction'] if has_toxic_prediction else nr_rows * [None],
    })
    return {
        'metrics': aggregate_metrics,
        'predictions': predictions,
    }


def evaluate_lexicon(
        lexicon_tokens, df, split='dev', propagate_binary_predictions=True, nr_spaces_to_fill=1,
):
    split = df[df['split'] == split]
    toxic_mask = split['toxic'].to_numpy()
    span_mask = split['toxic_mask'].map(lambda x: any(x)).to_numpy()
    span_predictions = []

    nr_rows = len(split)
    pct_predicted = np.full(nr_rows, np.nan)
    f1, p, r = np.full(nr_rows, np.nan), np.full(nr_rows, np.nan), np.full(nr_rows, np.nan)
    empty_pred, empty_label = np.full(nr_rows, False), np.full(nr_rows, False)
    for i, (_, row) in tqdm(enumerate(split.iterrows()), total=nr_rows, leave=False):
        full_text = row.full_text

        label = set(i for i, b in enumerate(row.toxic_mask) if b)
        pred = set()

        if not propagate_binary_predictions or row.prediction:
            # The binary model predicted toxic, use lexicon for span
            expr = "|".join([re.escape(token) for token in lexicon_tokens])
            for match in re.finditer(expr, full_text, re.I):
                pred.update(set(range(match.start(), match.end())))

            pred = _fill_empty_spaces(pred, nr_spaces_to_fill)

        span_predictions.append([True if i in pred else False for i in range(len(full_text))])
        f1[i], p[i], r[i], empty_result = metrics(pred, label)
        empty_pred[i] = MetricEmptySetResult.PRED in empty_result
        empty_label[i] = MetricEmptySetResult.LABEL in empty_result
        pct_predicted[i] = len(pred) / len(full_text) if len(full_text) > 0 else float('nan')

    aggr_metrics = _calc_aggregate_metrics(f1, p, r, toxic_mask, span_mask, empty_pred, empty_label, pct_predicted)
    aggr_metrics['lexicon_size'] = len(lexicon_tokens)

    predictions = pd.DataFrame({
        'sample_id': split['id'],
        'span_prediction': span_predictions,
        'toxic_prediction': toxic_mask,
    })
    return {
        'metrics': aggr_metrics,
        'predictions': predictions,
    }
