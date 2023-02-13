
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
    non_toxic_pct_predicted FLOAT,
    nr_empty_pred INT,
    nr_empty_label INT,
    nr_empty_both INT,
    nr_samples INT,
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
   f1_har_macro FLOAT GENERATED ALWAYS AS (CASE
        WHEN NOT (isnan(f1_toxic) OR isnan(f1_non_toxic))
        THEN 2 / ((1/f1_toxic) + (1/f1_non_toxic))
        ELSE 'NaN' END
    ) VIRTUAL,
    precision_har_macro FLOAT GENERATED ALWAYS AS (CASE
        WHEN NOT (isnan(precision_toxic) OR isnan(precision_non_toxic))
        THEN 2 / ((1/precision_toxic) + (1/precision_non_toxic))
        ELSE 'NaN' END
    ) VIRTUAL,
    recall_har_macro FLOAT GENERATED ALWAYS AS (CASE
        WHEN NOT (isnan(recall_toxic) OR isnan(recall_non_toxic))
        THEN 2 / ((1/recall_toxic) + (1/recall_non_toxic))
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

CREATE VIEW max_f1_toxic_view AS SELECT e_max.method_type, e_max_p.*, le.*, re.*, pe.*
FROM
(
	SELECT eval_dataset, train_dataset, MAX(f1_toxic) AS max_f1_toxic,
	CASE
		WHEN le.id IS NOT NULL THEN 'lexicon'
		WHEN pe.id IS NOT NULL THEN 'prediction'
		WHEN re.id IS NOT NULL THEN 'rationale'
		ELSE ''
	END AS method_type
	FROM evaluation AS e
	LEFT OUTER JOIN lexicon_evaluation AS le
	ON e.id = le.id
	LEFT OUTER JOIN prediction_evaluation AS pe
	ON e.id = pe.id
	LEFT OUTER JOIN rationale_evaluation AS re
	ON e.id = re.id
	GROUP BY method_type, eval_dataset, train_dataset
) AS e_max
LEFT OUTER JOIN evaluation AS e_max_p
ON e_max.eval_dataset = e_max_p.eval_dataset
AND e_max.train_dataset = e_max_p.train_dataset
AND e_max.max_f1_toxic = e_max_p.f1_toxic
LEFT OUTER JOIN lexicon_evaluation AS le
ON e_max_p.id = le.id
LEFT OUTER JOIN prediction_evaluation AS pe
ON e_max_p.id = pe.id
LEFT OUTER JOIN rationale_evaluation AS re
ON e_max_p.id = re.id;

CREATE VIEW max_f1_har_macro_view AS SELECT e_max.method_type, e_max_p.*, le.*, re.*, pe.*
FROM
(
	SELECT eval_dataset, train_dataset, MAX(f1_har_macro) AS max_f1_har_macro,
	CASE
		WHEN le.id IS NOT NULL THEN 'lexicon'
		WHEN pe.id IS NOT NULL THEN 'prediction'
		WHEN re.id IS NOT NULL THEN 'rationale'
		ELSE ''
	END AS method_type
	FROM evaluation AS e
	LEFT OUTER JOIN lexicon_evaluation AS le
	ON e.id = le.id
	LEFT OUTER JOIN prediction_evaluation AS pe
	ON e.id = pe.id
	LEFT OUTER JOIN rationale_evaluation AS re
	ON e.id = re.id
	GROUP BY method_type, eval_dataset, train_dataset
) AS e_max
LEFT OUTER JOIN evaluation AS e_max_p
ON e_max.eval_dataset = e_max_p.eval_dataset
AND e_max.train_dataset = e_max_p.train_dataset
AND e_max.max_f1_har_macro = e_max_p.f1_har_macro
LEFT OUTER JOIN lexicon_evaluation AS le
ON e_max_p.id = le.id
LEFT OUTER JOIN prediction_evaluation AS pe
ON e_max_p.id = pe.id
LEFT OUTER JOIN rationale_evaluation AS re
ON e_max_p.id = re.id;


