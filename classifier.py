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
            count_0 = 0
            count_1 = 0
            with open(self.single_stats) as csv_file:
                csv_reader = csv.reader(csv_file)
                row_counter = 0
                for row in csv_reader:
                    # Index counter for feature columns
                    # 0 = headline, 1 = class, 2,... = features
                    f_counter = 2
                    # First line contains column names
                    if row_counter == 0:
                        feature_names = row[f_counter:]
                        # Add column labels to class statistics csv
                        labels = ['is_sarcastic'] + feature_names
                        self.add_csv_entry(self.class_stats, labels)
                        # Class figures
                        # Value index 0 for class 0 (no sarcasm)
                        # Value index 1 for class 1 (sarcasm)
                        class_figs = {fn: [0, 0] for fn in feature_names}
                    # Nonsarcastic entry
                    elif int(row[1]) == 0:
                        count_0 += 1
                        for feature in feature_names:
                            class_figs[feature][0] += float(row[f_counter])
                            f_counter += 1
                    # Sarcastic entry
                    else:
                        count_1 += 1
                        for feature in feature_names:
                            class_figs[feature][1] += float(row[f_counter])
                            f_counter += 1
                    row_counter += 1
            stats_0 = [0] + [(class_figs[f][0]/count_0) for f in class_figs]
            stats_1 = [1] + [(class_figs[f][1]/count_1) for f in class_figs]
            self.add_csv_entry(self.class_stats, stats_0)
            self.add_csv_entry(self.class_stats, stats_1)

    def add_csv_entry(self, csv_file, new_entry):
        """Add entry to csv file"""
        with open(csv_file, mode='a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(new_entry)
        csv_file.close()

    def predict(self, test_data, test_stats_file, pred_csv):
        if os.path.exists(test_stats_file):
            logging.info('Statistics for prediction already computed. '
                         'Check if predictions already exist...')
            if os.path.exists(pred_csv):
                logging.info('Predictions found in {}'.format(pred_csv))
                return
            else:
                logging.info('No predictions found. Start predicting...')
        else:
            logging.info('Computing statistics for prediction data...')
            TestData = HeadlineData(test_data, test_stats_file)
            TestData.compute_single_statistics()
            self.add_csv_entry(pred_csv, ['headline', 'gold', 'prediction'])
            logging.info('Start predicting...')
        if os.path.exists(self.class_stats):
            with open(self.class_stats) as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    if row[0] == 'is_sarcastic':
                        continue
                    elif int(row[0]) == 0:
                        nonsarcastic_stats = row[1:]
                    elif int(row[0]) == 1:
                        sarcastic_stats = row[1:]
                    else:
                        logging.info('Wrong entries in class statistics')
                csv_file.close()
                with open(test_stats_file) as test_stats:
                    csv_reader = csv.reader(test_stats)
                    row_counter = 0
                    for row in csv_reader:
                        if row_counter == 0:
                            row_counter += 1
                            continue
                        stats_to_predict = row[2:]
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
                        self.add_csv_entry(pred_csv, [row[0],
                                                      row[1],
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