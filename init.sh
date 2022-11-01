# prepare data folder
rm -rf data/
mkdir -p data/cad \
         data/hatexplain \
         data/semeval

pushd data/
  pushd toxic-span/
    pushd cad/
      wget https://zenodo.org/record/4881008/files/data.zip?download=1 -O data.zip
      unzip data.zip
    popd

    pushd hatexplain/
      cp ../../data-repos/hatexplain/Data/dataset.json .
      cp ../../data-repos/hatexplain/Data/post_id_divisions.json .
    popd

    pushd semeval/
      cp ../../data-repos/semeval/SemEval2021/data/tsd_trial.csv .
      cp ../../data-repos/semeval/SemEval2021/data/tsd_test.csv .
      cp ../../data-repos/semeval/SemEval2021/data/tsd_train.csv .
    popd
  popd

  pushd toxic-lexicon/

  popd
popd

# install package locally
pip install -e .