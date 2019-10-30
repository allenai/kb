
KnowBert
========

KnowBert is a general method to embed multiple knowledge bases into BERT.
This repository contains pretrained models, evaluation and training scripts
for KnowBert with Wikipedia and WordNet.

Citation:

```
@inproceedings{Peters2019KnowledgeEC,
  author={Matthew E. Peters and Mark Neumann and Robert L Logan and Roy Schwartz and Vidur Joshi and Sameer Singh and Noah A. Smith},
  title={Knowledge Enhanced Contextual Word Representations},
  booktitle={EMNLP},
  year={2019}
}
```

## Getting started

```
git clone git@github.com:allenai/kb.git
cd kb
conda create -n knowbert python=3.6.7
source activate knowbert
pip install torch==1.2.0
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
python -m spacy download en_core_web_sm
python setup.py install
```

Then make sure the tests pass:

```
pytest -v tests
```


## Pretrained Models

* [KnowBert-WordNet](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wordnet_model.tar.gz)
* [KnowBert-Wiki](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_model.tar.gz)
* [KnowBert-W+W](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz)


## How to run intrinisic evaluation

First download one of the pretrained models from the previous section.

### Heldout perplexity (Table 1)

Download the [heldout data](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/data/wikipedia_bookscorpus_knowbert_heldout.txt). Then run:

```
MODEL_ARCHIVE=..location of model
HELDOUT_FILE=wikipedia_bookscorpus_knowbert_heldout.txt
python bin/evaluate_perplexity.py -m $MODEL_ARCHIVE -e $HELDOUT_FILE
```

The heldout perplexity is key `exp(lm_loss_wgt)`.

### Wikidata KG probe (Table 1)

Run:

```
MODEL_ARCHIVE=..location of model

mkdir -p kg_probe
cd kg_probe
curl https://allennlp.s3-us-west-2.amazonaws.com/knowbert/data/kg_probe.zip > kg_probe.zip
unzip kg_probe.zip

cd ..
python bin/evaluate_mrr.py \
    --model_archive $MODEL_ARCHIVE \
    --datadir kg_probe \
    --cuda_device 0
```

The results are in key `'mrr'`.


### Word-sense disambiguation

To evaluate the internal WordNet linker on the ALL task evaluation
from Navigli et al. (2017) follow these steps (Table 2).  First download the [Java scorer](http://lcl.uniroma1.it/wsdeval/) and [evaluation file](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wordnet/semeval2007_semeval2013_semeval2015_senseval2_senseval3_all.json).

Then run this command to generate predictions from KnowBert:

```
EVALUATION_FILE=semeval2007_semeval2013_semeval2015_senseval2_senseval3_all.json
KNOWBERT_PREDICTIONS=knowbert_wordnet_predicted.txt
MODEL_ARCHIVE=..location of model

python bin/evaluate_wsd_official.py \
    --evaluation_file $EVALUATION_FILE \
    --output_file $KNOWBERT_PREDICTIONS \
    --model_archive $MODEL_ARCHIVE \
    --cuda_device 0
```

To evaluate predictions, decompress the Java scorer, navigate to the directory `WSD_Evaluation_Framework/Evaluation_Datasets` and run

```
java Scorer ALL/ALL.gold.key.txt $KNOWBERT_PREDICTIONS
```

### AIDA Entity linking

To reproduce the results in Table 3 for KnowBert-W+W:

```
# or aida_test.txt
EVALUATION_FILE=aida_dev.txt
MODEL_ARCHIVE=..location of model

curl https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/$EVALUATION_FILE > $EVALUATION_FILE

python bin/evaluate_wiki_linking.py \
    --model_archive $MODEL_ARCHIVE \
    --evaluation_file $EVALUATION_FILE \
    --wiki_and_wordnet
```

Results are in key `wiki_el_f1`.


## Fine tuning KnowBert for downstream tasks

Fine tuning KnowBert is similar to fine tuning BERT for a downstream task.
We provide configuration and model files for the following tasks:

* Relation extraction: TACRED and SemEval 2010 Task 8
* Entity typing (Choi et al 2018)
* Binary sentence classification: Words-in-Context

To reproduce our results for the following tasks, find the appropriate config
file in `training_config/downstream/`, edit the location of the training and dev
data files, then run (example provided for TACRED):

```
allennlp train --file-friendly-logging --include-package kb.include_all \
        training_config/downstream/tacred.jsonnet -s OUTPUT_DIRECTORY
```

Similar to BERT, for some tasks performance can vary significantly with hyperparameter
choices and the random seed.  We used the script `bin/run_hyperparameter_seeds.sh`
to perform a small grid search over learning rate, number of epochs and the random seed,
choosing the best model based on the validation set.

### Evaluating fine tuned models

Fine-tuned KnowBert-Wiki+Wordnet models are available.

* [TACRED](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_tacred.tar.gz)
* [SemEval2010 Task 8](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_semeval2010_task8.tar.gz)
* [WiC](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_wic.tar.gz)
* [Entity Typing](https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_entity_typing.tar.gz)

To evaluate a model first download the model archive and run:

```
allennlp evaluate --include-package kb.include_all \
    --cuda-device 0 \
    model_archive_here \
    dev_or_test_filename_here
```

#### TACRED

To evaluate a model with the official scorer, run:

```
python bin/write_tacred_for_official_scorer.py \
    --model_archive model_archive_here \
    --evaluation_file tacred_dev_or_test.json \
    --output_file knowbert_predictions_tacred_dev_or_test.txt

python bin/tacred_scorer.py tacred_dev_or_test.gold knowbert_predictions_tacred_dev_or_test.txt
```

#### SemEval 2010 Task 8

To evaluate a model with the official scorer, first download
the [testing gold keys](https://github.com/teffland/Relation-Extraction/blob/master/SemEval2010_task8_all_data/test_keys.txt) and run:

```
curl https://allennlp.s3-us-west-2.amazonaws.com/knowbert/data/semeval2010_task8/test.json > semeval2010_task8_test.json

python bin/write_semeval2010_task8_for_official_eval.py \
    --model_archive model_archive_here \
    --evaluation_file semeval2010_task8_test.json \
    --output_file knowbert_predictions_semeval2010_task8_test.txt

perl -w bin/semeval2010_task8_scorer-v1.2.pl knowbert_predictions_semeval2010_task8_test.txt semeval2010_task8_testing_keys.txt
```

#### WiC

Use `bin/write_wic_for_codalab.py` to write a file for submission to the CodaLab evaluation server.



## How to pretrain KnowBert

Roughly speaking, the process to fine tune BERT into KnowBert is:

1. Prepare your corpus.
2. Prepare the knowledge bases (not necessary if you are using Wikipedia or WordNet as we have already prepared these).
3. For each knowledge base:
    1. Pretrain the entity linker while freezing everything else.
    2. Fine tune all parameters (except entity embeddings).


#### Prepare your corpus.
1. Sentence tokenize your training corpus using spacy, and prepare input files for next-sentence-prediction sampling.  Each file contains one sentence per line with consecutive sentences on subsequent lines and blank lines separating documents.
2. Run `bin/create_pretraining_data_for_bert.py` to group the sentences by length, do the NSP sampling, and write out files for training.
3. Reserve one or more of the training files for heldout evaluation.

#### Prepare the input knowledge bases.
1. We have already prepared the knowledge bases for Wikipedia and WordNet.  The necessary files will be automatically downloaded as needed when running evaluations or fine tuning KnowBert.
2. If you would like to add an additional knowledge source to KnowBert, these are roughly the steps to follow:

    1. Compute entity embeddings for each entity in your knowledge base.
    2. Write a candidate generator for the entity linkers.  Use the existing WordNet or Wikipedia generators as templates.

3.  Our Wikipedia candidate dictionary list and embeddings were extracted from [End-to-End Neural Entity Linking, Kolitsas et al 2018](https://github.com/dalab/end2end_neural_el) via a manual process.

4. Our WordNet candidate generator is rule based (see code).  The embeddings were computed via a multistep process that combines [TuckER](https://arxiv.org/abs/1901.09590) and [GenSen](https://github.com/Maluuba/gensen) embeddings.  The prepared files contain everything needed to run KnowBert and include:

    1. `entities.jsonl` - metadata about WordNet synsets.
    2. `wordnet_synsets_mask_null_vocab.txt` and `wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5` - vocabulary file and embedding file for WordNet synsets.
    3. `semcor_and_wordnet_examples.json` annotated training data combining SemCor and WordNet examples for supervising the WordNet linker.

5. If you would like to generate these files yourself from scratch, follow these steps.

   1. Extract the WordNet metadata and relationship graph.
        ```
        python bin/extract_wordnet.py --extract_graph --entity_file $WORKDIR/entities.jsonl --relationship_file $WORKDIR/relations.txt
        ```
    2. Download the [Words-in-Context dataset](https://pilehvar.github.io/wic/) to exclude from the extracted WordNet example usages.
        ```
        WORKDIR=.
        cd $WORKDIR
        wget https://pilehvar.github.io/wic/package/WiC_dataset.zip
        unzip WiC_dataset.zip
        ```
    2. Download the [word sense diambiguation data](http://lcl.uniroma1.it/wsdeval/):
        ```
        cd $WORKDIR
        wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
        unzip WSD_Evaluation_Framework.zip
        ```
    2. Convert the WSD data from XML to jsonl, and concatenate all evaluation files for easy evaluation:
        ```
        mkdir $WORKDIR/wsd_jsonl
        python bin/preprocess_wsd.py --wsd_framework_root $WORKDIR/WSD_Evaluation_Framework  --outdir $WORKDIR/wsd_jsonl
        cat $WORKDIR/wsd_jsonl/semeval* $WORKDIR/wsd_jsonl/senseval* > $WORKDIR/semeval2007_semeval2013_semeval2015_senseval2_senseval3.json
        ```
    2. Extract all the synset example usages from WordNet (after removing sentences from WiC heldout sets):
        ```
        python bin/extract_wordnet.py --extract_examples_wordnet --entity_file $WORKDIR/entities.jsonl --wic_root_dir $WORKDIR --wordnet_example_file $WORKDIR/wordnet_examples_remove_wic_devtest.json
        ```
    2. Combine WordNet examples and definitions with SemCor for training KnowBert:
        ```
        cat $WORKDIR/wordnet_examples_remove_wic_devtest.json $WORKDIR/wsd_jsonl/semcor.json > $WORKDIR/semcor_and_wordnet_examples.json
        ```
    3. Create training and test splits of the relationship graph.
        ```
        python bin/extract_wordnet.py --split_wordnet --relationship_file $WORKDIR/relations.txt --relationship_train_file $WORKDIR/relations_train99.txt --relationship_dev_file $WORKDIR/relations_dev01.txt
        ```
    4. Train TuckER embeddings on the extracted graph.  The configuration files uses relationship graph files on S3, although you can substitute them for the files generated in the previous step by modifying the configuration file.
        ```
        allennlp train -s $WORKDIR/wordnet_tucker --include-package kb.kg_embedding --file-friendly-logging training_config/wordnet_tucker.json
        ```
    5. Generate a vocabulary file useful for WordNet synsets with special tokens
        ```
        python bin/combine_wordnet_embeddings.py --generate_wordnet_synset_vocab --entity_file $WORKDIR/entities.jsonl --vocab_file $WORKDIR/wordnet_synsets_mask_null_vocab.txt
        ```
    6. Get the [GenSen](https://github.com/Maluuba/gensen) embeddings from each synset definition.  First install the code from this link.  Then run
        ```
        python bin/combine_wordnet_embeddings.py --generate_gensen_embeddings --entity_file $WORKDIR/entities.jsonl --vocab_file $WORKDIR/wordnet_synsets_mask_null_vocab.txt --gensen_file $WORKDIR/gensen_synsets.hdf5
        ```
    7. Extract the TuckER embeddings for the synsets from the trained model
        ```
        python bin/combine_wordnet_embeddings.py --extract_tucker --tucker_archive_file $WORKDIR/wordnet_tucker/model.tar.gz --vocab_file $WORKDIR/wordnet_synsets_mask_null_vocab.txt --tucker_hdf5_file $WORKDIR/tucker_embeddings.hdf5
        ```
    8. Finally combine the TuckER and GenSen embeddings into one file
        ```
        python bin/combine_wordnet_embeddings.py --combine_tucker_gensen --tucker_hdf5_file $WORKDIR/tucker_embeddings.hdf5 --gensen_file $WORKDIR/gensen_synsets.hdf5 --all_embeddings_file $WORKDIR/wordnet_synsets_mask_null_vocab_embeddings_tucker_gensen.hdf5
        ```

#### Pretraining the entity linkers

This step pretrains the entity linker while freezing the rest of the network using only supervised data.

Config files are in `training_config/pretraining/knowbert_wiki_linker.jsonnet` and `training_config/pretraining/knowbert_wordnet_linker.jsonnet`.

To train the Wikipedia linker for KnowBert-Wiki run:
```
allennlp train -s OUTPUT_DIRECTORY --file-friendly-logging --include-package kb.include_all training_config/pretraining/knowbert_wiki_linker.jsonnet
```

The command is similar for WordNet.

#### Fine tuning BERT

After pre-training the entity linkers from the step above, fine tune BERT.
The pretrained models in our paper were trained on a single GPU with 24GB of RAM.  For multiple GPU training, change `cuda_device` to a list of device IDs.

Config files are in `training_config/pretraining/knowbert_wiki.jsonnet` and
`training_config/pretraining/knowbert_wordnet.jsonnet`.

Before training, modify the following keys in the config file (or use `--overrides` flag to `allennlp train`):

* `"language_modeling"`
* `"model_archive"` to point to the `model.tar.gz` from the previous linker pretraining step.


#### KnowBert Wordnet + Wiki

First train KnowBert-Wiki.  Then pretrain the WordNet linker and finally fine tune the entire network.

Config file to pretrain the WordNet linker from KnowBert-Wiki is in `training_config/pretraining/knowbert_wordnet_wiki_linker.jsonnet` and config to train KnowBert-W+W is in `training_config/pretraining/knowbert_wordnet_wiki.jsonnet`.


