import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from nltk.stem import PorterStemmer

from toxic_x_dom.data import SPAN_DATASETS

DEFAULT_MODEL = LogisticRegression(max_iter=1000, class_weight='balanced')


def add_predictions_to_dataset(dataset_name, model=DEFAULT_MODEL):
    dataset = SPAN_DATASETS[dataset_name]()

    vectorizer_kwargs = dict(lowercase=True, stop_words='english', max_features=2000)
    analyzer = CountVectorizer(**vectorizer_kwargs).build_analyzer()
    stemmer = PorterStemmer()

    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))

    vectorizer = CountVectorizer(**vectorizer_kwargs, analyzer=stemmed_words)

    dataset['bag'] = vectorizer.fit_transform(dataset['full_text'].values).todense().tolist()

    train_split = dataset.loc[dataset['split'] == 'train']
    X, y = np.array(list(train_split['bag'].values)), np.array(list(train_split['toxic'].values))
    model.fit(X, y)

    dev_split = dataset.loc[dataset['split'] == 'dev']
    X = np.array(list(dev_split['bag'].values))
    predictions = model.predict(X).tolist()

    f1 = metrics.f1_score(dev_split['toxic'], predictions)

    dataset.loc[dataset['split'] == 'dev', 'prediction'] = predictions
    return dataset, f1

