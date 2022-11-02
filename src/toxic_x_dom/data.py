from types import MappingProxyType

import csv
import re
import json
from typing import Dict

import pandas as pd

CAD_TSV = 'data/toxic-span/cad/data/cad_v1_1.tsv'

SEMEVAL_CSVs = MappingProxyType({
    'train': 'data/toxic-span/semeval/tsd_train.csv',
    'trial': 'data/toxic-span/semeval/tsd_trial.csv',
    'test': 'data/toxic-span/semeval/tsd_test.csv',
})

HATEXPLAIN_JSONs = MappingProxyType({
    'data': 'data/toxic-span/hatexplain/dataset.json',
    'splits': 'data/toxic-span/hatexplain/post_id_divisions.json',
})


def load_cad_data(data_path: str = CAD_TSV):
    cad_df = pd.read_csv(
        data_path,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False
    )
    cad_df = cad_df.loc[cad_df['split'].isin(['train', 'dev', 'test'])]

    cad_df['toxic'] = cad_df.apply(lambda row: row.annotation_Primary != "Neutral", axis=1)

    cad_df = cad_df.rename(columns={
        "meta_text": "full_text",
    })

    def create_mask(toxic_text, full_text):
        if toxic_text == '"NA"':
            return [False for _ in range(len(full_text))]

        exp = r'\s+(\[linebreak\]\s*)*'.join(
            [re.escape(word) for word in toxic_text.split(' ')]
        )

        matches = list(re.finditer(exp, full_text, re.I))
        if len(matches) != 1:
            print(f'\nFound a toxic span with {len(matches)} matches.')
            print(full_text)
            print(exp)
            print(toxic_text)

        label = set()
        for match in matches:
            label |= set(range(match.start(), match.end()))

        # don't count '[linebreak]'s
        for match in re.finditer(re.escape('[linebreak]'), full_text):
            label -= set(range(match.start(), match.end()))

        mask = [i in label for i in range(len(full_text))]
        return mask

    cad_df['toxic_mask'] = cad_df.apply(
        lambda row: create_mask(row.annotation_highlighted, row.full_text),
        axis=1
    )

    def extract_tokens(toxic_text):
        if toxic_text == '"NA"':
            return []
        return [token.lower() for token in toxic_text.split(' ')]

    cad_df['toxic_tokens'] = cad_df.apply(
        lambda row: extract_tokens(row.annotation_highlighted), axis=1
    )

    cad_df = cad_df.drop(columns=['info_id', 'info_subreddit', 'info_subreddit_id',
                                  'info_id.parent', 'info_id.link', 'info_thread.id',
                                  'info_order', 'info_image.saved', 'annotation_Target',
                                  'annotation_Target_top.level.category',
                                  'annotation_Secondary', 'annotation_Context',
                                  'meta_author', 'meta_created_utc', 'meta_date', 'meta_day',
                                  'meta_permalink', 'subreddit_seen', 'annotation_Primary',
                                  'annotation_highlighted'])
    return cad_df


def load_hatexplain_data(data_paths=HATEXPLAIN_JSONs):
    hxpl_df = pd.read_json(data_paths['data'], orient="index")
    hxpl_df = hxpl_df.rename(columns={"post_id": "id"})

    hxpl_df['toxic'] = hxpl_df.apply(lambda row: row.rationales != [], axis=1)
    hxpl_df['full_text'] = hxpl_df.apply(
        lambda row: " ".join(row.post_tokens),
        axis=1
    )

    def convert_rationales(rationales, full_text):
        if len(rationales) == 0:
            return [False for _ in range(len(full_text))]
        rationale = [
            sum(r[i] for r in rationales) > len(rationales) / 2.
            for i in range(len(rationales[0]))
        ]
        return rationale

    hxpl_df['rationale'] = hxpl_df.apply(
        lambda row: convert_rationales(row.rationales, row.full_text), axis=1)

    def create_mask(rationale, post_tokens):
        # assume whitespace is not offensive
        return sum([[v] * len(w) + [False] for v, w in zip(rationale, post_tokens)], [])[:-1]

    hxpl_df['toxic_mask'] = hxpl_df.apply(
        lambda row: create_mask(row.rationale, row.post_tokens), axis=1)

    hxpl_df['toxic_tokens'] = hxpl_df.apply(
        lambda row: [
            token.lower() for token, r in zip(row.post_tokens, row.rationale) if r
        ], axis=1
    )

    hxpl_df = hxpl_df.drop(columns=['annotators', 'rationales', 'rationale',
                                    'post_tokens'])

    with open(data_paths['splits']) as json_file:
        hex_splits = json.load(json_file)

    splits = {'train': 'train', 'val': 'dev', 'test': 'test'}
    for split_key, ids in hex_splits.items():
        hxpl_df.loc[hxpl_df['id'].isin(ids), 'split'] = splits[split_key]

    return hxpl_df


def load_semeval_data(data_paths=SEMEVAL_CSVs):
    sem_df_train = pd.read_csv(data_paths['train'], quotechar='"', keep_default_na=False, header=0)
    sem_df_trial = pd.read_csv(data_paths['trial'], quotechar='"', keep_default_na=False, header=0)
    sem_df_test = pd.read_csv(data_paths['test'], quotechar='"', keep_default_na=False, header=0)
    sem_df_train['split'] = 'train'
    sem_df_trial['split'] = 'dev'
    sem_df_test['split'] = 'test'
    sem_df = pd.concat((sem_df_train, sem_df_trial, sem_df_test))

    sem_df = sem_df.rename(columns={"text": "full_text"})

    def convert_spans(spans_str):
        if spans_str == "[]":
            return []
        return list(map(int, spans_str.strip('][').split(', ')))

    sem_df['spans'] = sem_df.apply(lambda row: convert_spans(row.spans), axis=1)
    sem_df['id'] = sem_df.apply(lambda row: f"{row.split}.{row.name}", axis=1)
    sem_df['toxic'] = sem_df.apply(lambda row: row.spans != [], axis=1)
    sem_df['toxic_mask'] = sem_df.apply(lambda row: [
        i in row.spans for i, _ in enumerate(row.full_text)], axis=1)

    def extract_toxic_tokens(full_text, spans):
        toxic_tokens = []

        def add_phrase(phrase):
            toxic_tokens.extend([token.lower() for token in phrase.split(' ')])

        phrase = ""
        for i, s_i in enumerate(spans):
            s_i = int(s_i)
            phrase += full_text[s_i]
            if i + 1 == len(spans):
                add_phrase(phrase)
                break
            s_j = int(spans[i + 1])

            if s_i + 1 != s_j:
                add_phrase(phrase)
                phrase = ""
        return toxic_tokens

    sem_df['toxic_tokens'] = sem_df.apply(
        lambda row: extract_toxic_tokens(row.full_text, row.spans), axis=1)

    sem_df = sem_df.drop(columns=['spans'])
    return sem_df


def load_toxic_span_datasets(
        cad_data_path=CAD_TSV,
        hatexplain_data_paths=HATEXPLAIN_JSONs,
        semeval_data_paths=SEMEVAL_CSVs,
):
    return {
        'cad': load_cad_data(data_path=cad_data_path),
        'semeval': load_semeval_data(data_paths=semeval_data_paths),
        'hatexplain': load_hatexplain_data(data_paths=hatexplain_data_paths)
    }


HURTLEX_TSV = 'data/toxic-lexicon/hurtlex/hurtlex_EN.tsv'

WIEGAND_TXTs = MappingProxyType({
    'base': 'data/toxic-lexicon/wiegand/baseLexicon.txt',
    'expanded': 'data/toxic-lexicon/wiegand/expandedLexicon.txt',
})


def load_hurtlex(data_path: str = HURTLEX_TSV):
    lexicon = pd.read_csv(data_path, sep='\t')
    return {
        'hurtlex-cons': set(lexicon[lexicon['level'] == 'conservative']['lemma']),
        'hurtlex-incl': set(lexicon['lemma']),
    }


def load_wiegand(data_paths: Dict[str, str] = WIEGAND_TXTs):
    base = pd.read_csv(data_paths['base'], sep='\t', names=['word_pos', 'abusive'])
    expanded = pd.read_csv(data_paths['expanded'], sep='\t', names=['word_pos', 'score'])
    base[['word', 'pos']] = base.apply(lambda row: row.word_pos.split('_'), axis=1, result_type='expand')
    expanded[['word', 'pos']] = expanded.apply(lambda row: row.word_pos.split('_'), axis=1, result_type='expand')

    return {
        'wiegand-base': set(base[base['abusive']]['word']),
        'wiegand-expa': set(expanded[expanded['score'] > 0]['word']),
    }


def load_lexicons(hurtlex_data_path=HURTLEX_TSV, wiegand_data_paths=WIEGAND_TXTs):
    return load_hurtlex(hurtlex_data_path) | load_wiegand(wiegand_data_paths)


if __name__ == "__main__":
    def print_sample(df):
        print("\nTOXIC")
        print(df.loc[df['toxic']].head(8))
        print("\nNON-TOXIC")
        print(df.loc[~df['toxic']].head(8))

    print("Sample from CAD data.")
    print_sample(load_cad_data())

    print("Sample from HateXplain data.")
    print_sample(load_hatexplain_data())

    print("Sample from Semeval data.")
    print_sample(load_semeval_data())
