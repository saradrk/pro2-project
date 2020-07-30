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
        self.datafile = datafile
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
            TrainingData = HeadlineData(self.datafile, self.single_stats)
            TrainingData.compute_single_statistics()
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

    def add_csv_entry(self, csv_file, new_entry):
        """Add entry to csv file"""
        with open(csv_file, mode='a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(new_entry)
        csv_file.close()

    def predict(self, test_data, test_stats_file, pred_csv):
        TestData = HeadlineData(test_data, test_stats_file)
        TestData.compute_single_statistics()
        self.add_csv_entry(pred_csv, ['headline', 'gold', 'prediction'])
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
                with open(test_stats_file) as test_stats:
                    csv_reader = csv.reader(test_stats)
                    for row in csv_reader:
                        stats_to_predict = row[3:]
                        nonsarcastic_dist = self._distance(nonsarcastic_stats,
                                                           stats_to_predict)
                        sarcastic_dist = self._distance(sarcastic_stats,
                                                        stats_to_predict)
                        if nonsarcastic_dist < sarcastic_dist:
                            prediction = 0
                        elif nonsarcastic_dist > sarcastic_dist:
                            prediction = 1
                        else:
                            prediction = 0.5
                        self.add_csv_entry(pred_csv, [row[1],
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


if __name__ == '__main__':
    start_time = time.time()
    C = Classifier('./csv/mini_single_stats.csv',
                   './csv/mini_class_stats.csv',
                   './Data/mini_data.json'
                   )
    C.train_model()
    C.predict('./Data/mini_test.json',
              './csv/mini_test_single_stats.csv',
              './csv/mini_test_predictions.csv')
    duration = time.time() - start_time
    logging.info(f'Running time classifier: {duration} sec')