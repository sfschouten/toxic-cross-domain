from collections import Counter

import math

import argparse
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def count_tokens(df, non_neutral_only=False, minimum_occurrences=0):
    whole_corpus = Counter()
    in_abusive = Counter()

    train_split = df.loc[df['split'] == 'train']
    for index, row in train_split.iterrows():
        toxic_tokens = row.toxic_tokens
        full_text = row.full_text

        def remove_punctuation(tokens):
            return [token for token in tokens if token.isalpha()]

        text_tokens = word_tokenize(full_text.lower())
        text_tokens = remove_punctuation(text_tokens)

        whole_corpus.update(text_tokens if not non_neutral_only else toxic_tokens)

        if not row.toxic:
            continue

        toxic_tokens = remove_punctuation(toxic_tokens)

        # only update in_abusive with tokens that are also in the full text
        # (sometimes the highlight begins/ends mid-word)
        in_both = set(toxic_tokens) & set(text_tokens)
        toxic_tokens = [t for t in toxic_tokens if t in in_both]
        in_abusive.update(toxic_tokens)

    tokens = set(in_abusive.keys()) & set(whole_corpus.keys())

    if minimum_occurrences > 0:
        too_low = set(key for key, count in whole_corpus.items() if count < minimum_occurrences)
        tokens -= too_low

    return in_abusive, whole_corpus, tokens


def calculate_scores(in_abusive, whole_corpus, tokens, z=0):
    def toxic_score(token):
        a = in_abusive[token]
        n = whole_corpus[token]
        score = a / n

        # calculate prior probability of a token being highlighted
        p = sum(in_abusive.values()) / sum(whole_corpus.values())
        q = 1 - p

        # Some tokens only appear once or twice, so prevalence among highlights
        # has large error bars. We can use low bar as score (configurable by
        # `z` parameter, use z=0 to use raw score).
        low_bar = z * math.sqrt(p * q / n)
        return score - low_bar

    scores = {token: toxic_score(token) for token in tokens}
    return scores


def construct_lexicon(scores, theta=0.5):
    result = sorted(
        filter(lambda x: x[1] > theta, scores.items()),
        key=lambda x: x[1], reverse=True
    )
    return result


if __name__ == "__main__":
    from toxic_x_dom.data import SPAN_DATASETS

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=['cad', 'semeval', 'hatexplain'], default='hatexplain')

    parser.add_argument('--min_occurrence', default=7, type=int)
    parser.add_argument('--join_predicted', choices=[True, False], default=True, type=bool)

    parser.add_argument('--theta', default=0.3, type=float)

    args = parser.parse_args()

    df = SPAN_DATASETS[args.dataset]()
    _in_abusive, _whole_corpus, _tokens = count_tokens(
        df, non_neutral_only=False, minimum_occurrences=args.min_occurrence
    )

    scores_ = calculate_scores(_in_abusive, _whole_corpus, _tokens)
    lexicon = construct_lexicon(scores_, theta=args.theta)

    import csv

    with open('lexicon.csv', 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['word', 'toxicity'])
        csv_out.writerows(lexicon)
