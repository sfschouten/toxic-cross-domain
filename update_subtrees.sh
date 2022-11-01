#!/usr/bin/env bash

# add dataset repos as remotes
git remote add -f cad_repo  https://github.com/dongpng/cad_naacl2021.git
git remote add -f hatexplain_repo https://github.com/hate-alert/HateXplain.git
git remote add -f semeval_repo https://github.com/ipavlopoulos/toxic_spans.git


# subtree the repos
git subtree add --prefix data/cad cad_repo main --squash
git subtree add --prefix data/hatexplain hatexplain_repo master --squash
git subtree add --prefix data/semeval semeval_repo master --squash


#