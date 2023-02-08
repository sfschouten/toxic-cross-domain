
CREATE TABLE evaluation(
    id UUID PRIMARY KEY DEFAULT uuid(),
    eval_dataset VARCHAR,
    train_dataset VARCHAR,
    propagate_binary BOOLEAN,
    filling_chars INT,
    f1_micro FLOAT,
    precision_micro FLOAT,
    recall_micro FLOAT,
    f1_toxic FLOAT,
    precision_toxic FLOAT,
    recall_toxic FLOAT,
    f1_toxic_no_span FLOAT,
    precision_toxic_no_span FLOAT,
    recall_toxic_no_span FLOAT,
    f1_non_toxic FLOAT,
    precision_non_toxic FLOAT,
    recall_non_toxic FLOAT,
    f1_macro FLOAT GENERATED ALWAYS AS (CASE
        WHEN NOT (isnan(f1_toxic) OR isnan(f1_non_toxic))
        THEN (f1_toxic + f1_non_toxic) / 2
        ELSE 'NaN' END
    ) VIRTUAL,
    precision_macro FLOAT GENERATED ALWAYS AS (CASE
        WHEN NOT (isnan(precision_toxic) OR isnan(precision_non_toxic))
        THEN (precision_toxic + precision_non_toxic) / 2
        ELSE 'NaN' END
    ) VIRTUAL,
    recall_macro FLOAT GENERATED ALWAYS AS (CASE
        WHEN NOT (isnan(recall_toxic) OR isnan(recall_non_toxic))
        THEN (recall_toxic + recall_non_toxic) / 2
        ELSE 'NaN' END
    ) VIRTUAL,
);

CREATE TABLE lexicon_evaluation(
    id UUID PRIMARY KEY,
    lexicon_size INT,
    min_occurrence INT,
    theta FLOAT,
    FOREIGN KEY (id) REFERENCES evaluation(id)
);

CREATE TABLE rationale_evaluation(
    id UUID PRIMARY KEY,
    attribution_method VARCHAR,
    scale_scores BOOLEAN,
    cumulative_scoring BOOLEAN,
    threshold FLOAT,
    FOREIGN KEY (id) REFERENCES evaluation(id)
);

CREATE TABLE prediction_evaluation(
    id UUID PRIMARY KEY,
    FOREIGN KEY (id) REFERENCES evaluation(id)
);