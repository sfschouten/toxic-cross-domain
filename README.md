# Cross-domain toxic span prediction
Project to evaluate toxic span prediction methods in a cross-domain setting.

### Directories 
- `data-repos`:  copies (with `git subtree`) of the repositories for each of the datasets we use.
- `src/toxic_x_dom/`: files with source code relevant for all approaches, and a submodule for 
each approach.
- `experiments/`: configurations used to obtain the experimental results.

### Getting started
To start using this code run `init.sh` (after creating a virtualenv or conda environment).
This will copy the relevant data into a `data` folder, and then locally install this python package. 

Dependencies are listed in `environment.yml`.

### Reproducing results
Use scripts in `experiments/` to train models.

Example: `./run.sh MODEL_DATASET.json`
