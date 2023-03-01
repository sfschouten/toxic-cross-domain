import sys
import os
import csv
import re
import json
import logging
from dataclasses import dataclass
from typing import Dict
from types import MappingProxyType
from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import datasets
from datasets import DownloadManager, Dataset, load_dataset

from tqdm import tqdm

load_dotenv()

PROJECT_HOME = os.getenv('TOXIC_X_DOM_HOME')

CAD_TSV = os.path.join(PROJECT_HOME, 'data/toxic-span/cad/data/cad_v1_1.tsv')

SEMEVAL_CSVs = MappingProxyType({
    'train': os.path.join(PROJECT_HOME, 'data/toxic-span/semeval/tsd_train.csv'),
    'trial': os.path.join(PROJECT_HOME, 'data/toxic-span/semeval/tsd_trial.csv'),
    'test': os.path.join(PROJECT_HOME, 'data/toxic-span/semeval/tsd_test.csv'),
    'civil_comments_train': os.path.join(PROJECT_HOME, 'data/toxic-span/semeval/civil_comments/train.csv'),
})

HATEXPLAIN_JSONs = MappingProxyType({
    'data': os.path.join(PROJECT_HOME, 'data/toxic-span/hatexplain/dataset.json'),
    'splits': os.path.join(PROJECT_HOME, 'data/toxic-span/hatexplain/post_id_divisions.json'),

})

HURTLEX_TSV = os.path.join(PROJECT_HOME, 'data/toxic-lexicon/hurtlex/hurtlex_EN.tsv')

WIEGAND_TXTs = MappingProxyType({
    'base': os.path.join(PROJECT_HOME, 'data/toxic-lexicon/wiegand/baseLexicon.txt'),
    'expanded': os.path.join(PROJECT_HOME, 'data/toxic-lexicon/wiegand/expandedLexicon.txt'),
})


def load_cad_data(data_path: str = CAD_TSV, remove_non_hateful_slurs=True):
    cad_df = pd.read_csv(
        data_path,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False
    )
    cad_df = cad_df.loc[cad_df['split'].isin(['train', 'dev', 'test'])]

    non_hateful_slur_str = 'Slur'
    if remove_non_hateful_slurs:
        cad_df = cad_df[cad_df.annotation_Primary != non_hateful_slur_str]
    non_toxic_strs = ["Neutral", "CounterSpeech", non_hateful_slur_str]
    cad_df['toxic'] = cad_df.apply(lambda row: row.annotation_Primary not in non_toxic_strs, axis=1)

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
            logging.debug(f' Found a toxic span with {len(matches)} matches.'
                          f'\nFULL TEXT:\n{full_text}'
                          f'\nREGEX:\n{exp}'
                          f'\nTOXIC TEXT:\n{toxic_text}\n')

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

    def is_toxic(annotators):
        counts = Counter([annotator['label'] for annotator in annotators])
        return counts['offensive'] + counts['hatespeech'] > counts['normal']

    hxpl_df['toxic'] = hxpl_df.apply(lambda row: is_toxic(row.annotators), axis=1)
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


def load_semeval_data(data_paths=SEMEVAL_CSVs, ratio_nontoxic=1., civil_comments_sample_seed=0):
    sem_df_train = pd.read_csv(data_paths['train'], quotechar='"', keep_default_na=False, header=0)
    sem_df_trial = pd.read_csv(data_paths['trial'], quotechar='"', keep_default_na=False, header=0)
    sem_df_test = pd.read_csv(data_paths['test'], quotechar='"', keep_default_na=False, header=0)
    sem_df_train['split'] = 'train'
    sem_df_trial['split'] = 'dev'
    sem_df_test['split'] = 'test'
    sem_df = pd.concat((sem_df_train, sem_df_trial, sem_df_test))
    len_semeval = len(sem_df_train), len(sem_df_trial), len(sem_df_test)

    sem_df = sem_df.rename(columns={"text": "full_text"})
    del sem_df_train, sem_df_trial, sem_df_test

    def convert_spans(spans_str):
        if spans_str == "[]":
            return []
        return list(map(int, spans_str.strip('][').split(', ')))

    sem_df['spans'] = sem_df.apply(lambda row: convert_spans(row.spans), axis=1)
    sem_df['id'] = sem_df.apply(lambda row: f"{row.split}.{row.name}", axis=1)
    sem_df['toxic'] = True
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

    if ratio_nontoxic > 0:
        # Supplement with non-toxic samples from civil comments
        civil_df = pd.concat([chunk for chunk in tqdm(
            pd.read_csv(
                data_paths['civil_comments_train'], chunksize=1000, quotechar='"', keep_default_na=False, header=0
            ), desc='Loading Civil Comments data'
        )])

        # Drop unnecessary columns
        civil_df = civil_df.drop(
            columns=civil_df.columns.difference(['id', 'target', 'comment_text', 'toxicity_annotator_count'])
        )
        civil_df = civil_df.rename(columns={'comment_text': 'full_text'})

        # Require at least 3 annotators, at least half of whom reported no toxicity
        # (Pavlopoulos et al. require the same when selecting for toxic samples).
        civil_df = civil_df.loc[(civil_df['target'] < 0.5) & (civil_df['toxicity_annotator_count'] > 3)]

        # Select appropriate number of samples given the desired ratio.
        nr_nontoxic = tuple(length * ratio_nontoxic for length in len_semeval)
        if sum(nr_nontoxic) <= len(civil_df):
            logging.info(f'Drawing {int(sum(nr_nontoxic))} non-toxic samples from total of {len(civil_df)}.')
            civil_df = civil_df.sample(n=int(sum(nr_nontoxic)), random_state=civil_comments_sample_seed)
        else:
            raise ValueError('Not enough data in Civil Comments to get desired ratio.')

        civil_df['id'] = civil_df.apply(lambda row: f"cc.{row.id}", axis=1)
        civil_df['toxic'] = False
        civil_df['toxic_mask'] = civil_df.apply(lambda row: [False] * len(row.full_text), axis=1)
        civil_df['toxic_tokens'] = [[] for _ in range(len(civil_df))]

        s = np.cumsum(nr_nontoxic, dtype=np.int)
        civil_df.loc[civil_df.index[:s[0]],     'split'] = 'train'
        civil_df.loc[civil_df.index[s[0]:s[1]], 'split'] = 'dev'
        civil_df.loc[civil_df.index[s[1]:],     'split'] = 'test'
        civil_df = civil_df.drop(columns=['target', 'toxicity_annotator_count'])
        sem_df = pd.concat((sem_df,  civil_df))

    return sem_df


SPAN_DATASETS = {
    'cad': load_cad_data,
    'semeval': load_semeval_data,
    'hatexplain': load_hatexplain_data
}


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


@dataclass()
class ToxicSpanBuilderConfig(datasets.BuilderConfig):
    dataset_name: str = None

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Loading an Toxic Span dataset requires you specify "
                             "the name of the dataset you want to load.")


class ToxicSpanDatasetBuilder(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = ToxicSpanBuilderConfig
    NAME_TO_FUNCTION = {
        'cad': load_cad_data,
        'semeval': load_semeval_data,
        'hatexplain': load_hatexplain_data
    }

    def _info(self) -> datasets.DatasetInfo:
        dataset_name = self.config.dataset_name
        self.dataset = self.NAME_TO_FUNCTION[dataset_name]()
        all_data_hf = Dataset.from_pandas(self.dataset)
        return all_data_hf.info

    def _split_generators(self, dl_manager: DownloadManager):
        return [
            datasets.SplitGenerator(name=split_key, gen_kwargs={'split_name': split_key})
            for split_key in ['train', 'dev', 'test']
        ]

    def _generate_tables(self, split_name):
        yield 0, Dataset.from_pandas(self.dataset.loc[self.dataset['split'] == split_name]).data.table


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
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def print_sample(df, name):
        print(f"Sample from {name} data.")
        print("\nTOXIC")
        print(df.loc[df['toxic']].head(8))
        print("\nNON-TOXIC")
        print(df.loc[~df['toxic']].head(8))

    def print_crosstab_toxic_nospan(df):
        # print how many samples labelled toxic do not have spans
        has_span = df.apply(lambda row: np.array(row.toxic_mask).any(), axis=1)
        is_toxic =  df['toxic']
        split = df['split']
        crosstab = pd.crosstab(has_span, [split, is_toxic],
                               rownames=['Has span?'], colnames=['Split', 'Is toxic?'])
        print(crosstab)

    def print_span_proportions(df):
        lengths = df.apply(lambda row: len(row.full_text), axis=1)[df['toxic']]
        nr_toxic = df.apply(lambda row: sum(row.toxic_mask), axis=1)[df['toxic']]
        avg_proportion = (nr_toxic[lengths > 0] / lengths[lengths > 0]).mean()
        print(f'average proportion of sample that is toxic: {avg_proportion}')

    DATASETS = {'CAD': load_cad_data, 'HateXplain': load_hatexplain_data, 'SemEval': load_semeval_data}
    for key, data_fn in DATASETS.items():
        data = data_fn()
        print_sample(data, key)
        print_crosstab_toxic_nospan(data)
        print(print_span_proportions(data))
