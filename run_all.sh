#!/bin/bash

pushd experiments/
  pushd rationale_extraction/
    ./run.sh bert_cad.json
    ./run.sh bert_semeval.json
    ./run.sh bert_hatexplain.json
    ./run_test.sh bert_cad.json
    ./run_test.sh bert_semeval.json
    ./run_test.sh bert_hatexplain.json
  popd
  pushd span_detection/grid/
    ./run.sh bert_cad.json
    ./run.sh bert_semeval.json
    ./run.sh bert_hatexplain.json
    ./run.sh bert_crf_cad.json
    ./run.sh bert_crf_semeval.json
    ./run.sh bert_crf_hatexplain.json
    ./run_test.sh bert_cad.json
    ./run_test.sh bert_semeval.json
    ./run_test.sh bert_hatexplain.json
    ./run_test.sh bert_crf_cad.json
    ./run_test.sh bert_crf_semeval.json
    ./run_test.sh bert_crf_hatexplain.json
  popd
  pushd lexicon/
    ./run.sh
    ./run_test.sh
  popd
popd
