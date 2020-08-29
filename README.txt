Irony classifier for newspaper headlines

This repository consists of:
	classifier.py
		Implements Classifier class with train, predict and accuracy method.
		Uses HeadlineData class for processing the data.
	process_data.py
		Implements HeadlineData class for processing the data and extracting
		linguistic features of the data corpus for classification.
		Uses Headline class for feature extraction.
	headline.py 
		Implements Headline class for natural language processing and linguistic
		feature extraction of single headlines.
	split_data.py
		Splits data into training/test/validation sets.
	main.py
		Instantiates Classifier object. Trains on training data
		(Sarcasm_Headlines_Dataset_v2_train.json), Predicts testing data
		(Sarcasm_Headlines_Dataset_v2_test.json) and returns accuracy.
	set-up.sh
		Creates 'Data' and 'csv' directory, splits data and installs requirements.
		(Further instructions for usage under 1: Preparations)
	requirements.txt 
		Contains requirements for running the classifier.

1: Preparations
----------------

* Download Version 2 of the data (Sarcasm_Headlines_Dataset_v2.json) from:
https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
* Unpack the data and move it to this folder (the project folder)
* In the following step you will download several python modules,
so preferably you should create a virtual environment by running:
	python3 -m venv env
Activate it either bei running:
	./env/bin/activate
or
	source ./env/bin/activate
(depending on your operating system)
* Now run 'set-up.sh' from this folder
You should now find 3 new folders in this directory ('env', 'csv', 'Data').
In the 'Data' folder you should find following files:
	Sarcasm_Headlines_Dataset_v2_val.json
	Sarcasm_Headlines_Dataset_v2_test.json
	Sarcasm_Headlines_Dataset_v2_train.json
	Sarcasm_Headlines_Dataset_v2.json

2: Demonstration of the classifier
----------------------------------

To see a demonstration of the classifier you 

