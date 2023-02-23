
CREATE TABLE evaluation(
    id UUID PRIMARY KEY DEFAULT uuid(),
    timestamp TIMESTAMP DEFAULT current_timestamp,
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
    lexicon_key VARCHAR,
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

CREATE TABLE span_pred_evaluation(
    id UUID PRIMARY KEY,
    FOREIGN KEY (id) REFERENCES evaluation(id)
);

CREATE TABLE span_data(
    sample_id VARCHAR PRIMARY KEY,
    dataset_key VARCHAR,
    split VARCHAR,
    full_text VARCHAR,
    toxic BOOLEAN,
    toxic_mask BOOLEAN[],
    toxic_tokens VARCHAR[],
);

CREATE TABLE predictions(
    evaluation_id UUID,
    sample_id VARCHAR,
    toxic_prediction BOOLEAN,
    span_prediction BOOLEAN[],
    PRIMARY KEY (evaluation_id, sample_id),
    FOREIGN KEY (evaluation_id) REFERENCES evaluation(id),
);


CREATE VIEW all_max_f1_toxic_view AS SELECT e_max.method_type, e_max_p.*, le.*, re.*, pe.*
FROM
(
	SELECT eval_dataset, train_dataset, MAX(f1_toxic) AS max_f1_toxic,
	CASE
		WHEN le.id IS NOT NULL THEN 'lexicon'
		WHEN pe.id IS NOT NULL THEN 'span_pred'
		WHEN re.id IS NOT NULL THEN 'rationale'
		ELSE ''
	END AS method_type
	FROM evaluation AS e
	LEFT OUTER JOIN lexicon_evaluation AS le ON e.id = le.id
	LEFT OUTER JOIN span_pred_evaluation AS pe	ON e.id = pe.id
	LEFT OUTER JOIN rationale_evaluation AS re ON e.id = re.id
	GROUP BY method_type, eval_dataset, train_dataset
) AS e_max
LEFT OUTER JOIN evaluation AS e_max_p
ON e_max.eval_dataset = e_max_p.eval_dataset
AND e_max.train_dataset = e_max_p.train_dataset
AND e_max.max_f1_toxic = e_max_p.f1_toxic
LEFT OUTER JOIN lexicon_evaluation AS le ON e_max_p.id = le.id
LEFT OUTER JOIN span_pred_evaluation AS pe ON e_max_p.id = pe.id
LEFT OUTER JOIN rationale_evaluation AS re ON e_max_p.id = re.id;


CREATE VIEW all_max_f1_har_macro_view AS SELECT e_max.method_type, e_max_p.*, le.*, re.*, pe.*
FROM
(
	SELECT eval_dataset, train_dataset, MAX(f1_har_macro) AS max_f1_har_macro,
	CASE
		WHEN le.id IS NOT NULL THEN 'lexicon'
		WHEN pe.id IS NOT NULL THEN 'span_pred'
		WHEN re.id IS NOT NULL THEN 'rationale'
		ELSE ''
	END AS method_type
	FROM evaluation AS e
	LEFT OUTER JOIN lexicon_evaluation AS le ON e.id = le.id
	LEFT OUTER JOIN span_pred_evaluation AS pe ON e.id = pe.id
	LEFT OUTER JOIN rationale_evaluation AS re ON e.id = re.id
	GROUP BY method_type, eval_dataset, train_dataset
) AS e_max
LEFT OUTER JOIN evaluation AS e_max_p
ON e_max.eval_dataset = e_max_p.eval_dataset
AND e_max.train_dataset = e_max_p.train_dataset
AND e_max.max_f1_har_macro = e_max_p.f1_har_macro
LEFT OUTER JOIN lexicon_evaluation AS le ON e_max_p.id = le.id
LEFT OUTER JOIN span_pred_evaluation AS pe ON e_max_p.id = pe.id
LEFT OUTER JOIN rationale_evaluation AS re ON e_max_p.id = re.id;


CREATE VIEW tuned_in_domain_max_f1_har_macro_view AS
	WITH evaluation_plus AS (
	    SELECT *,
	    CASE
	        WHEN le.id IS NOT NULL THEN 'lexicon'
	        WHEN pe.id IS NOT NULL THEN 'span_pred'
	        WHEN re.id IS NOT NULL THEN 'rationale'
	        ELSE ''
	    END AS method_type
	    FROM evaluation AS e
	    LEFT OUTER JOIN lexicon_evaluation AS le ON e.id = le.id
	    LEFT OUTER JOIN span_pred_evaluation AS pe ON e.id = pe.id
	    LEFT OUTER JOIN rationale_evaluation AS re ON e.id = re.id
	), in_domain_max AS (
	    SELECT method_type, train_dataset, lexicon_key, attribution_method, propagate_binary, eval_dataset, MAX(f1_har_macro) AS max_f1_har_macro
	    FROM evaluation_plus
	    WHERE train_dataset = eval_dataset
	    GROUP BY method_type, train_dataset, lexicon_key, attribution_method, propagate_binary, eval_dataset
	), in_domain_max_ids AS (
	    SELECT evaluation_plus.id
	    FROM in_domain_max
	    LEFT OUTER JOIN evaluation_plus
	    ON in_domain_max.method_type = evaluation_plus.method_type
	    AND in_domain_max.train_dataset = evaluation_plus.train_dataset
	    AND in_domain_max.lexicon_key IS NOT DISTINCT FROM evaluation_plus.lexicon_key
	    AND in_domain_max.attribution_method IS NOT DISTINCT FROM evaluation_plus.attribution_method
	    AND in_domain_max.propagate_binary = evaluation_plus.propagate_binary
	    AND in_domain_max.eval_dataset = evaluation_plus.eval_dataset
	    AND in_domain_max.max_f1_har_macro = evaluation_plus.f1_har_macro
	), cross_domain_max AS (
	    SELECT e_cross.*
	    FROM in_domain_max_ids AS e
	    LEFT OUTER JOIN evaluation_plus AS e_in
	    ON e.id = e_in.id
	    LEFT OUTER JOIN evaluation_plus AS e_cross
	    ON  e_in.propagate_binary IS NOT DISTINCT FROM e_cross.propagate_binary
	    AND e_in.filling_chars IS NOT DISTINCT FROM e_cross.filling_chars
	    AND e_in.train_dataset = e_cross.train_dataset
	    AND (
	        (
	            e_in.method_type = 'lexicon' AND e_in.method_type = e_cross.method_type
	            AND e_in.lexicon_key IS NOT DISTINCT FROM e_cross.lexicon_key
	            AND e_in.theta IS NOT DISTINCT FROM e_cross.theta
	            AND e_in.min_occurrence IS NOT DISTINCT FROM e_cross.min_occurrence
	        ) OR (
	            e_in.method_type = 'rationale' AND e_in.method_type = e_cross.method_type
	            AND e_in.attribution_method IS NOT DISTINCT FROM e_cross.attribution_method
	            AND e_in.scale_scores IS NOT DISTINCT FROM e_cross.scale_scores
	            AND e_in.cumulative_scoring IS NOT DISTINCT FROM e_cross.cumulative_scoring
	            AND e_in.threshold IS NOT DISTINCT FROM e_cross.threshold
	        ) OR (
	            e_in.method_type = 'span_pred' AND e_in.method_type = e_cross.method_type
	        )
	    )
	)
	SELECT * FROM cross_domain_max;


CREATE VIEW tuned_in_domain_max_f1_toxic_view AS
	WITH evaluation_plus AS (
	    SELECT *,
	    CASE
	        WHEN le.id IS NOT NULL THEN 'lexicon'
	        WHEN pe.id IS NOT NULL THEN 'span_pred'
	        WHEN re.id IS NOT NULL THEN 'rationale'
	        ELSE ''
	    END AS method_type
	    FROM evaluation AS e
	    LEFT OUTER JOIN lexicon_evaluation AS le ON e.id = le.id
	    LEFT OUTER JOIN span_pred_evaluation AS pe ON e.id = pe.id
	    LEFT OUTER JOIN rationale_evaluation AS re ON e.id = re.id
	), in_domain_max AS (
	    SELECT method_type, train_dataset, lexicon_key, attribution_method, propagate_binary, eval_dataset, MAX(f1_toxic) AS max_f1_toxic
	    FROM evaluation_plus
	    WHERE train_dataset = eval_dataset
	    GROUP BY method_type, train_dataset, lexicon_key, attribution_method, propagate_binary, eval_dataset
	), in_domain_max_ids AS (
	    SELECT evaluation_plus.id
	    FROM in_domain_max
	    LEFT OUTER JOIN evaluation_plus
	    ON in_domain_max.method_type = evaluation_plus.method_type
	    AND in_domain_max.train_dataset = evaluation_plus.train_dataset
	    AND in_domain_max.lexicon_key IS NOT DISTINCT FROM evaluation_plus.lexicon_key
	    AND in_domain_max.attribution_method IS NOT DISTINCT FROM evaluation_plus.attribution_method
	    AND in_domain_max.propagate_binary = evaluation_plus.propagate_binary
	    AND in_domain_max.eval_dataset = evaluation_plus.eval_dataset
	    AND in_domain_max.max_f1_toxic = evaluation_plus.f1_toxic
	), cross_domain_max AS (
	    SELECT e_cross.*
	    FROM in_domain_max_ids AS e
	    LEFT OUTER JOIN evaluation_plus AS e_in
	    ON e.id = e_in.id
	    LEFT OUTER JOIN evaluation_plus AS e_cross
	    ON  e_in.propagate_binary IS NOT DISTINCT FROM e_cross.propagate_binary
	    AND e_in.filling_chars IS NOT DISTINCT FROM e_cross.filling_chars
	    AND e_in.train_dataset = e_cross.train_dataset
	    AND (
	        (
	            e_in.method_type = 'lexicon' AND e_in.method_type = e_cross.method_type
	            AND e_in.lexicon_key IS NOT DISTINCT FROM e_cross.lexicon_key
	            AND e_in.theta IS NOT DISTINCT FROM e_cross.theta
	            AND e_in.min_occurrence IS NOT DISTINCT FROM e_cross.min_occurrence
	        ) OR (
	            e_in.method_type = 'rationale' AND e_in.method_type = e_cross.method_type
	            AND e_in.attribution_method IS NOT DISTINCT FROM e_cross.attribution_method
	            AND e_in.scale_scores IS NOT DISTINCT FROM e_cross.scale_scores
	            AND e_in.cumulative_scoring IS NOT DISTINCT FROM e_cross.cumulative_scoring
	            AND e_in.threshold IS NOT DISTINCT FROM e_cross.threshold
	        ) OR (
	            e_in.method_type = 'span_pred' AND e_in.method_type = e_cross.method_type
	        )
	    )
	)
	SELECT * FROM cross_domain_max;
