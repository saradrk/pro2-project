31.08.2020 - Pro2-A SoSe2020: Final Assignment - Sara Derakhshani


Irony classifier for newspaper headlines
________________________________________


This repository consists of:

	README.txt
		This file.
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
	test.py 
		Unittest cases.
	/test_data
		Contains 'pred_test_data.json' and 'test_data.json' for unit testing.
	set-up.sh
		Creates 'Data' directory, splits data and installs requirements.
		(Further instructions for usage under 1: Preparations)
	requirements.txt 
		Contains requirements for running the classifier.
	

1: Preparations
---------------

Instructions after downloading the repository:
* Download Version 2 of the data (Sarcasm_Headlines_Dataset_v2.json) from:
https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
* Unpack the data and move it to this folder (the project folder).
* The data file should be named Sarcasm_Headlines_Dataset_v2.json. 
If this is not the case check if you really downloaded version 2 and/or change the name. 
* Open your shell and change the directory to the project folder.
* In the following step you will download several python modules,
so preferably you should create a virtual environment by running:
	python3 -m venv env
Activate it either bei running:
	./env/bin/activate
or
	source ./env/bin/activate
(depending on your operating system)
* Now run 'set-up.sh' from this folder.
You should now find 3 new folders in this directory ('env', 'csv', 'Data').
In the 'Data' folder you should find following files:
	Sarcasm_Headlines_Dataset_v2_val.json
	Sarcasm_Headlines_Dataset_v2_test.json
	Sarcasm_Headlines_Dataset_v2_train.json
	Sarcasm_Headlines_Dataset_v2.json


2: Demonstration of the classifier
----------------------------------

To see a demonstration of the classifier run 'main.py'.
It trains the classifier on the train split part of the Sarcasm Headline Dataset,
predicts the test split part and computes the accuracy of those predictions. 
The outputs are in 'irony_classifier.log' which is created with the first use of
the program.

To use the classifier with different train oder prediction set run 'classifier.py'.
It takes two positional arguments, the first one is the file name of the training set,
the second one the file name of the dataset for prediction. Predicting the validation set e.g.:

	python3 classifier.py Data/Sarcasm_Headlines_Dataset_v2_train.json Data/Sarcasm_Headlines_Dataset_v2_val.json

If the the classifier has already been trained on the training set and the csv files still exist,
the classifier does not train again and the existing files are used as classification model.


3: Unittests
------------

Run 'test.py'.


4: Using the Classifier class in a program
------------------------------------------

To use the classifier in a program first instantiate it with the file path to a dataset file (JSON).
After that use the train method which needs no arguments. 
Then you can use the predict method. It needs also a file path to a dataset file (JSON) as argument. 
After training on a dataset you can predict different datasets without training the model again.
The prediction method returns the file name that contains the predictions.
This output can be used as input for the accuracy method which needs the path to the predictions as argument.




 

