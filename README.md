# Exercise Logs for Coco Nutritionist

### processCSVforMajorityVote.py 
Uses the csv file from our qualtrics survey and creates a csv file used in majorityVote.py.

To run processCSVforMajorityVote.py: `python3 processCSVforMajorityVote.py`
(You must replace the name of the csv file on line 19 with the csv file you want to process)


### processCSVforCRF.py 
Uses the csv file from our qualtrics survey and creates a csv file that is used in crf.py and CRFwithPOS.py.

To run processCSVforCRF.py: `python3 processCSVforCRF.py`
(You must replace the name of the csv files on lines 15 and 16 with the csv file you want to process and write to)


### crf.py 
Uses the csv from processCSVforCRF.py and uses a sklearn Conditional Random Field adapted from a [tutorial](https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2) by Susan Li.

To run crf.py: `python3 crf.py`


### CRFwithPOS.py 
Uses the csv from processCSVforCRF.py and uses sklearn Conditional Random Field adapted from a [tutorial](https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2) by Susan Li.

To run CRFwithPOS.py: `python3 CRFwithPOS.py`
