# Exercise Logs for Coco Nutritionist

By Maya Epps under the mentorship of Dr. Mandy Korpusik

## Processing Qualtrics CSV:

### processCSVforMajorityVote.py 
Uses the csv file from our qualtrics survey and creates a csv file used in majorityVote.py.

To run processCSVforMajorityVote.py: `python3 processCSVforMajorityVote.py`
(You must replace the name of the csv file on line 19 with the csv file you want to process)


### processCSVforCRF.py 
Uses the csv file from our qualtrics survey and creates a csv file that is used in crf.py and CRFwithPOS.py.

To run processCSVforCRF.py: `python3 processCSVforCRF.py`
(You must replace the name of the csv files on lines 15 and 16 with the csv files you want to process and write to)


## Baselines:

### majorityVote.py
Uses the csv from processCSVforMajorityVote.py and creates a dictionary of the most common sematic tag ("majority vote") 

To run majorityVote.py: `python3 majorityVote.py`


### crf.py 
Uses the csv from processCSVforCRF.py and uses a sklearn Conditional Random Field adapted from a [tutorial](https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2) by Susan Li.

To run crf.py: `python3 crf.py`


### CRFwithPOS.py 
Uses the csv from processCSVforCRF.py and uses sklearn Conditional Random Field adapted from a [tutorial](https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2) by Susan Li. Uses part of speech (POS) tagging from Spacy.

To run CRFwithPOS.py: `python3 CRFwithPOS.py`

## Neural networks:

### nnets.py
Trains and tests one of the following neural networks: FF, CNN, LSTM

`python nnets.py`

## Contextual embeddings:

### bert.py
Uses the Transformers named entity recognition example on exercise logs. To use this, run the following:

`source venv/bin/activate`

`python3 preprocess.py train.txt.tmp bert-base-cased 150 > train.txt`

`python3 preprocess.py dev.txt.tmp bert-base-cased 150 > dev.txt`

`python3 preprocess.py test.txt.tmp bert-base-cased 150 > test.txt`

`python3 bert.py --data_dir ./ --labels ./labels.txt --output_dir models/bert-base-cased --max_seq_length 150 --num_train_epochs 3 --per_gpu_train_batch_size 32 --save_steps 750 --seed 1 --do_train --do_eval --do_predict --model_type bert --model_name_or_path bert-base-cased`

where `bert-base-cased` is just one possible model. Here are the model options:

`--model_type [bert, roberta, xlnet, albert, electra]`

`--model_name_or_path [bert-base-cased, bert-base-uncased, bert-large-cased, bert-large-uncased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, roberta-base, roberta-large, xlnet-base-cased, xlnet-large-cased, albert-base-v1, albert-large-v1, albert-xlarge-v1, albert-xxlarge-v1, albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2, google/electra-small-discriminator, google-electra-base-discriminator, google-electra-large-discriminator]`

For evaluation only, remove the `--do_train argument`.

## Intent detection:

### intentClassifier.py

Predicts the intent (i.e., either food or exercise logging) using statistical classifiers: random forest and logistic regression.

`python intentClassifier.py`