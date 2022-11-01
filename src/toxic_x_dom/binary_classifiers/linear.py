import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn import metrics


DEFAULT_MODEL = LogisticRegression(max_iter=1000, class_weight='balanced')


def add_predictions_to_datasets(dataset, model=DEFAULT_MODEL):
    vectorizer = CountVectorizer(lowercase=True, stop_words='english', max_features=2000)
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

