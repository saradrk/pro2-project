# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Extract features
# Store statistics

from process_data import HeadlineData
import csv
import os
import logging
import time

logging.basicConfig(filename='classifier.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


class Classifier:

    def __init__(self, single_stats_file, class_stats_file, datafile):
        self.single_stats = single_stats_file
        self.class_stats = class_stats_file
        self.headlines = HeadlineData(datafile)
        self.stats = {'awl': [0, 0],
                      'stop_words': [0, 0]}

    def train_model(self):
        """Create classification model as csv file containing statistics
        for each class
        Compute class statistics from single statistics in csv file format
        """
        # Test if model has been trained already
        if os.path.exists(self.single_stats):
            logging.info('Model is trained and ready to predict')
        # Sum up the figures from each feature in the single statistics for
        # the sarcastic and nonsarcastic class and divide by class count to
        # compute the mean figure for each feature
        # Create csv file to save class statistics
        else:
            self._compute_single_statistics()
            sarcasm_count = 0
            nonsarcasm_count = 0
            with open(self.single_stats) as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    f_counter = 3
                    if int(row[2]) == 0:
                        nonsarcasm_count += 1
                        for feature in self.stats:
                            self.stats[feature][0] += float(row[f_counter])
                            f_counter += 1
                    else:
                        sarcasm_count += 1
                        for feature in self.stats:
                            self.stats[feature][1] += float(row[f_counter])
                            f_counter += 1
            stats_0 = [0] + [(self.stats[f][0]/nonsarcasm_count) for f in self.stats]
            stats_1 = [1] + [(self.stats[f][1]/sarcasm_count) for f in self.stats]
            self.add_csv_entry(self.class_stats, stats_0)
            self.add_csv_entry(self.class_stats, stats_1)

    def _compute_single_statistics(self, pred_csv=None):
        if pred_csv is not None:
            for feature in self.headlines.features:
                self.add_csv_entry(pred_csv, feature)
            logging.info('Single statistics for predictions computed')
        else:
            for feature in self.headlines.features:
                self.add_csv_entry(self.single_stats, feature)
            logging.info('Single statistics computed')

    def add_csv_entry(self, csv_file, new_entry):
        """Add entry to csv file"""
        with open(csv_file, mode='a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(new_entry)
        csv_file.close()

    def predict(self):
        self._compute_single_statistics(pred_csv='stats_to_predict.csv')
        self.add_csv_entry('test_predictions.csv', ['headline', 'gold', 'prediction'])
        if os.path.exists(self.class_stats):
            with open(self.class_stats) as csv_file:
                csv_reader = csv.reader(csv_file)
                nonsarcastic_stats = []
                sarcastic_stats = []
                for row in csv_reader:
                    if int(row[0]) == 0:
                        nonsarcastic_stats += row[1:]
                    else:
                        sarcastic_stats += row[1:]
                csv_file.close()
                with open('stats_to_predict.csv') as csv_file_pred:
                    csv_reader = csv.reader(csv_file_pred)
                    for row in csv_reader:
                        stats_to_predict = row[3:]
                        if self._distance(nonsarcastic_stats, stats_to_predict) < self._distance(sarcastic_stats, stats_to_predict):
                            prediction = 0
                        elif self._distance(nonsarcastic_stats, stats_to_predict) > self._distance(sarcastic_stats, stats_to_predict):
                            prediction = 1
                        else:
                            prediction = 0.5
                        self.add_csv_entry('test_predictions.csv', [row[1],
                                                               row[2],
                                                               prediction]
                                           )
        else:
            logging.info('Model has not been trained yet')

    def _distance(self, trained_values, test_values):
        assert(len(trained_values) == len(test_values))
        dist = 0
        for i in range(len(trained_values)):
            feature_dist = (float(trained_values[i]) - float(test_values[i]))
            dist += abs(feature_dist)
        return dist

def accuracy(self, pred_csv):
    if os.path.exists(self.pred_csv):
        with open(self.pred_csv) as csv_file:
            csv_reader = csv.reader(csv_file)
            total_pred = -1
            correct_pred = 0
            for row in csv_reader:
                if total_pred == -1:
                    total_pred += 1
                else:
                    total_pred += 1
                    if row[1] == row[2]:
                        correct_pred += 1
        return (correct_pred/total_pred)

if __name__ == '__main__':
    start_time = time.time()
    C = Classifier('train_single_stats.csv',
               'train_class_stats.csv',
               './Data/Sarcasm_Headlines_Dataset_v2_val.json')
#    C.train_model()
#    C.predict()
    duration = time.time() - start_time
    logging.info(f'Running time predicting: {duration} sec')