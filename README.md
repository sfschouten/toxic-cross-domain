# Cross-domain toxic span prediction
Project to evaluate toxic span prediction methods in a cross-domain setting.

### Directories 
The `data-repos` folder has copies (with `git subtree`) of the repositories for each of the datasets we use.

In the `src/toxic_x_dom/` directory there are files with source code relevant for all approaches, and a submodule for 
each approach.

### Getting started
To start using this code run `init.sh` (after creating a virtualenv or conda environment).
This will copy the relevant data into a `data` folder, and then locally install this python package. 

Dependencies are listed in `environment.yml`.
