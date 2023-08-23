# Cross-domain toxic span prediction
Project to evaluate toxic span prediction methods in a cross-domain setting.

### Directories 
- `data-repos/`:  copies (with `git subtree`) of the repositories for each of the datasets we use.
- `src/toxic_x_dom/`: files with source code relevant for all approaches, and a submodule for 
each approach.
- `experiments/`: configurations used to obtain the experimental results.

### Getting started
To start using this code run `init.sh` (after creating a virtualenv or conda environment).
This will copy the relevant data into a `data/` folder, and then locally install this python package. 

Dependencies are listed in `environment.yml`.

### Reproducing results
Use scripts in `experiments/` to train models.

Example: `./run.sh MODEL_DATASET.json`



## Methods

### Lexicon-based
Implementation in `src/toxic_x_dom/lexicon/`.

This method involves the use of lexicons for the prediction of toxic spans.
Given an input text, the words that appear in the lexicon are marked as toxic.

We use both existing lexicons and lexicons generated from data annotated with toxic spans.

Generating the lexicons involves calculating a toxicity score for each word in the data by counting how often it occurs inside and outside toxic spans.
All words with a toxicity score above a threshold are considered toxic and added to the lexicon.
Mostly based on what was described by [Zhu et al. (2021)](https://aclanthology.org/2021.semeval-1.63/)

### Rationale extraction
Implementation in `src/toxic_x_dom/rationale_extraction/`.

This method uses input attribution methods to predict toxic spans.
We train a binary toxicity classifier, and then use input attribution methods to obtain a rationale for the prediction.
The rationale consists of an attribution score for each input token, and the toxic spans are predicted by thresholding the attribution scores.
Mostly based on what was described by [Pluciński et al. (2021)](https://aclanthology.org/2021.semeval-1.114/).

For an overview of which input attribution methods are implemented, see `src/toxic_x_dom/rationale_extraction/attribution.py`.


### Fine-tuned language models
Implementation in `src/toxic_x_dom/span_prediction/`.

This method involves fine-tuning a language model to predict toxic spans directly.
Currently, we use BERT models (with or without a CRF layer on top), to predict a BIO tag for each input token.


## Evaluation
Each method is evaluated cross-domain, in various settings:
 - assuming different levels of toxicity information
   - evaluated on only toxic text, evaluating purely the ability to predict toxic spans
   - evaluated on toxic and non-toxic text, including a binary toxicity prediction step from which errors are propagated
 - post-processing of the predictions
   - merging predicted spans separated by *x* tokens.
  
  
# Citation
Schouten, S.F., Barbarestani, B., Tufa, W., Vossen, P., Markov, I. (2023). Cross-Domain Toxic Spans Detection. In: Métais, E., Meziane, F., Sugumaran, V., Manning, W., Reiff-Marganiec, S. (eds) Natural Language Processing and Information Systems. NLDB 2023. Lecture Notes in Computer Science, vol 13913. Springer, Cham. https://doi.org/10.1007/978-3-031-35320-8_40
```bibtex
@InProceedings{10.1007/978-3-031-35320-8_40,
 author="Schouten, Stefan F.
 and Barbarestani, Baran
 and Tufa, Wondimagegnhue
 and Vossen, Piek
 and Markov, Ilia",
 editor="M{\'e}tais, Elisabeth
 and Meziane, Farid
 and Sugumaran, Vijayan
 and Manning, Warren
 and Reiff-Marganiec, Stephan",
 title="Cross-Domain Toxic Spans Detection",
 booktitle="Natural Language Processing and Information Systems",
 year="2023",
 publisher="Springer Nature Switzerland",
 address="Cham",
 pages="533--545",
 abstract="Given the dynamic nature of toxic language use, automated methods for detecting toxic spans are likely to encounter distributional shift. To explore this phenomenon, we evaluate three approaches for detecting toxic spans under cross-domain conditions: lexicon-based, rationale extraction, and fine-tuned language models. Our findings indicate that a simple method using off-the-shelf lexicons performs best in the cross-domain setup. The cross-domain error analysis suggests that (1) rationale extraction methods are prone to false negatives, while (2) language models, despite performing best for the in-domain case, recall fewer explicitly toxic words than lexicons and are prone to certain types of false positives. Our code is publicly available at: https://github.com/sfschouten/toxic-cross-domain.",
 isbn="978-3-031-35320-8"
}
