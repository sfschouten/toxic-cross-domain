#!/usr/bin/env bash

# add dataset repos as remotes
git remote add -f cad_repo  https://github.com/dongpng/cad_naacl2021.git
git remote add -f hatexplain_repo https://github.com/hate-alert/HateXplain.git
git remote add -f semeval_repo https://github.com/ipavlopoulos/toxic_spans.git

git remote add -f hurtlex_repo https://github.com/valeriobasile/hurtlex.git
git remote add -f wiegand_repo https://github.com/uds-lsv/lexicon-of-abusive-words.git


# pull the latest version of the repos
git subtree pull --prefix data-repos/cad cad_repo main --squash
git subtree pull --prefix data-repos/hatexplain hatexplain_repo master --squash
git subtree pull --prefix data-repos/semeval semeval_repo master --squash

git subtree pull --prefix data-repos/hurtlex hurtlex_repo main --squash
git subtree pull --prefix data-repos/wiegand wiegand_repo main --squash