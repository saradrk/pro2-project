# Sara Derakhshani
# 30.07.2020
# Programmierung II: Projekt
# Evaluate

import logging
import os
import csv

logging.basicConfig(filename='process_data.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


class Evaluator:

    def __init__(self, prediction_csv):
        self.predictions = prediction_csv

    def accuracy(self):
        if os.path.exists(self.predictions):
            with open(self.predictions) as predictions:
                csv_reader = csv.reader(predictions)
                total_pred = -1
                correct_pred = 0
                for pred in csv_reader:
                    if total_pred == -1:
                        total_pred += 1
                    else:
                        total_pred += 1
                        if pred[1] == pred[2]:
                            correct_pred += 1
            return (correct_pred/total_pred)


if __name__ == '__main__':
    Eval = Evaluator('test_predictions.csv')
    logging.info('Accuracy of classifier is: {}'.format(Eval.accuracy()))
