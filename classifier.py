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
    """Class for training a classifier for binary classification.

    Attributes:
        single_stats (str): name of csv file for single statistics of the data
        class_stats (str): name of csv file for class statistics
        datafile (str): name of csv file with headline data

    Methods:
        train_model(): compute class statistics for prediction
        predict(test_data, test_stats_file, pred_csv): classify unknown data
    """

    def __init__(self, single_stats_file, class_stats_file, datafile):
        """Constructor for Classifier class.

        Args:
            single_stats (str): name of csv file for single statistics
            class_stats (str): name of csv file for class statistics
            datafile (str): name of csv file with headline data
        """
        self.single_stats = single_stats_file
        self.class_stats = class_stats_file
        self.datafile = datafile

    def train_model(self):
        """Create classification model.

        Compute single statistics for data entries and save as csv file.
        Compute class statistics from single statistics and save as csv file.
        """
        # Test if model has been trained already
        if (os.path.exists(self.single_stats) and
                os.path.getsize(self.single_stats) != 0):
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
                        self._add_csv_entry(self.class_stats, labels)
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
            self._add_csv_entry(self.class_stats, stats_0)
            self._add_csv_entry(self.class_stats, stats_1)

    def _add_csv_entry(self, csv_file, new_entry):
        """Add entry to csv file.

        Args:
            csv_file (str): name of csv file entry should be added to
            new_entry (list): list containing column entries for new csv row
        """
        with open(csv_file, mode='a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(new_entry)
        csv_file.close()

    def predict(self, test_data, test_stats_file, pred_csv):
        """Classify data based on previously trained model.

        Args:
            test_data (str): name of csv file containing data to be classified
            test_stats_file (str): name of csv file to save single statistics
                of data to be classified
            pred_csv (str): name of csv file to save predictions in
        """
        # If single statistics of the data to predict already
        # exists start predicting (if predictions already made log info)
        # compute single statistics and predict otherwise
        if (os.path.exists(test_stats_file) and
                os.path.getsize(test_stats_file) != 0):
            logging.info('Statistics for prediction already computed. '
                         'Check if predictions already exist...')
            if (os.path.exists(pred_csv) and
                    os.path.getsize(pred_csv) != 0):
                logging.info('Predictions found in {}'.format(pred_csv))
                return None
            else:
                logging.info('No predictions found. Start predicting...')
        else:
            logging.info('Computing statistics for prediction data...')
            TestData = HeadlineData(test_data, test_stats_file)
            TestData.compute_single_statistics()
            self._add_csv_entry(pred_csv, ['headline', 'gold', 'prediction'])
            logging.info('Start predicting...')
        # Start predicting if model has been trained
        if (os.path.exists(self.class_stats) and
                os.path.getsize(self.class_stats) != 0):
            with open(self.class_stats) as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    # first row contains feature labels of training data
                    if row[0] == 'is_sarcastic':
                        all_features = row[1:]
                    # is_sarcastic == 0: non sarcastic statistics
                    elif int(row[0]) == 0:
                        ns_stats = row[1:]
                    # is_sarcastic == 1: sarcastic statistics
                    elif int(row[0]) == 1:
                        s_stats = row[1:]
                    else:
                        logging.info('Wrong entries in class statistics')
                csv_file.close()
                with open(test_stats_file) as test_stats:
                    csv_reader = csv.reader(test_stats)
                    row_counter = 0
                    eval_features = []
                    for row in csv_reader:
                        # first row contains feature labels of prediction data
                        if row_counter == 0:
                            eval_features += row[2:]
                            # get indices of relevant features for prediction
                            # in case more features were extracted for
                            # training data or features of prediction and
                            # training data are in different order
                            rel_ind = self._get_relevant_indices(all_features,
                                                                 eval_features)
                            # extract relevant class statistics
                            nonsarcastic_stats = [ns_stats[i] for i in rel_ind]
                            sarcastic_stats = [s_stats[i] for i in rel_ind]
                            # log features of prediction data
                            logging.info('Features used for prediction: '
                                         '{}'.format(eval_features))
                            row_counter += 1
                            continue
                        # compute distances between statistics of the
                        # prediction data entry and class entries
                        stats_to_predict = row[2:]
                        nonsarcastic_dist = self._distance(nonsarcastic_stats,
                                                           stats_to_predict)
                        sarcastic_dist = self._distance(sarcastic_stats,
                                                        stats_to_predict)
                        # Classify as the class with smaller distance
                        if nonsarcastic_dist < sarcastic_dist:
                            prediction = 0
                        elif nonsarcastic_dist > sarcastic_dist:
                            prediction = 1
                        else:
                            prediction = 0.5
                        self._add_csv_entry(pred_csv, [row[0],
                                                       row[1],
                                                       prediction
                                                       ]
                                            )
        else:
            logging.info('Model is not trained')

    def _get_relevant_indices(self, all_features, rel_features):
        """Return list of indices of the features relevant for classification.

        Args:
            all_features (list of strings): list with names of features
                extracted from training data
            rel_features (list of strings): list with names of features
                extracted from prediction data
        """
        feature_indeces = []
        for rel_f in rel_features:
            index_counter = 0
            for f in all_features:
                if rel_f == f:
                    feature_indeces.append(index_counter)
                    break
                else:
                    index_counter += 1
        return feature_indeces

    def _distance(self, trained_values, test_values):
        """Compute and return feature distance.

        Args:
            trained_values (list): list of feature statistics of a class
            test_values (list): list of feature statistics of a data instance
        """
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