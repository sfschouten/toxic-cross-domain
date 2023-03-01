import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from nltk.stem import PorterStemmer

from toxic_x_dom.data import SPAN_DATASETS

DEFAULT_MODEL = LogisticRegression(max_iter=1000, class_weight='balanced')


def add_predictions_to_dataset(eval_dataset_name, train_dataset_name, model=DEFAULT_MODEL, split_key='dev'):
    # TODO use split_key...
    raise NotImplementedError('Temporarily disable linear model until it is made compatible with cross-domain application')

    eval_dataset = SPAN_DATASETS[eval_dataset_name]()
    train_dataset = SPAN_DATASETS[train_dataset_name]()

    vectorizer_kwargs = dict(lowercase=True, stop_words='english', max_features=2000)
    analyzer = CountVectorizer(**vectorizer_kwargs).build_analyzer()
    stemmer = PorterStemmer()

    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))

    vectorizer = CountVectorizer(**vectorizer_kwargs, analyzer=stemmed_words)

    train_dataset['bag'] = vectorizer.fit_transform(train_dataset['full_text'].values).todense().tolist()
    eval_dataset['bag'] = vectorizer.fit_transform(eval_dataset['full_text'].values).todense().tolist()

    train_split = train_dataset.loc[train_dataset['split'] == 'train']
    X, y = np.array(list(train_split['bag'].values)), np.array(list(train_split['toxic'].values))
    model.fit(X, y)

    # evaluate
    dev_split = eval_dataset.loc[eval_dataset['split'] == 'dev']
    X = np.array(list(dev_split['bag'].values))
    predictions = model.predict(X).tolist()

    f1 = metrics.f1_score(dev_split['toxic'], predictions)

    eval_dataset.loc[eval_dataset['split'] == 'dev', 'prediction'] = predictions
    return eval_dataset, f1

