# prepare data folder
rm -rf data/
mkdir -p data/toxic-span/cad \
         data/toxic-span/hatexplain \
         data/toxic-span/semeval \
         data/toxic-lexicon/hurtlex \
         data/toxic-lexicon/wiegand \

pushd data/
  pushd toxic-span/
    pushd cad/
      curl https://zenodo.org/record/4881008/files/data.zip?download=1 -o data.zip
      unzip data.zip
    popd

    pushd hatexplain/
      cp ../../../data-repos/hatexplain/Data/dataset.json .
      cp ../../../data-repos/hatexplain/Data/post_id_divisions.json .
    popd

    pushd semeval/
      cp ../../../data-repos/semeval/SemEval2021/data/tsd_trial.csv .
      cp ../../../data-repos/semeval/SemEval2021/data/tsd_test.csv .
      cp ../../../data-repos/semeval/SemEval2021/data/tsd_train.csv .
    popd
  popd

  pushd toxic-lexicon/
    pushd hurtlex/
      cp ../../../data-repos/hurtlex/lexica/EN/1.2/hurtlex_EN.tsv .
    popd

    pushd wiegand/
      cp ../../../data-repos/wiegand/Lexicons/baseLexicon.txt .
      cp ../../../data-repos/wiegand/Lexicons/expandedLexicon.txt .
    popd
  popd
popd

# install package locally
pip install -e .
