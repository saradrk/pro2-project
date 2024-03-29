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
import argparse

logging.basicConfig(filename='irony_classifier.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


class Classifier:
    """Class for training a classifier for binary classification.

    Attributes:
        datafile_path (str): complete path to training data file
        path (str) path where the training data lives
        datafile (str): name of JSON file containing the training data
        single_stats (str): name of csv file for single statistics of the data
        class_stats (str): name of csv file for class statistics

    Methods:
        train_model(): compute class statistics for prediction
        predict(pred_datafile_path): classify data
        accuracy(pred_csv_path): compute prediction accuracy of classifier
    """

    def __init__(self, datafile_path):
        """Constructor for Classifier class.

        Args:
            datafile_path (str): complete path to training data file
        """
        self.datafile_path = datafile_path
        self.path = os.path.dirname(datafile_path)
        self.datafile = os.path.basename(datafile_path)
        single_stats_csv = self.datafile[:-5] + '_single_stats.csv'
        self.single_stats = os.path.join(self.path, single_stats_csv)
        class_stats_csv = self.datafile[:-5] + '_class_stats.csv'
        self.class_stats = os.path.join(self.path, class_stats_csv)

    def train_model(self):
        """Create classification model.

        Compute single statistics for data entries and save as csv file.
        Compute class statistics from single statistics and save as csv file.

        Return:
            Path to file containing computed class statistics (str)
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
            logging.info('Start training...')
            TrainingData = HeadlineData(self.datafile_path)
            TrainingData.compute_single_statistics(self.single_stats)
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
            logging.info('Training completed.')
            return self.class_stats

    def predict(self, pred_datafile_path):
        """Classify data based on previously trained model.

        Args:
            pred_datafile (str): name of JSON file containing data to classify
        Return:
            Path to csv file containing the predictions (str)
        """
        pred_data = os.path.basename(pred_datafile_path)
        pred_single_stats_csv = pred_data[:-5] + '_single_stats.csv'
        pred_single_stats = os.path.join(self.path, pred_single_stats_csv)
        pred_out_csv = pred_data[:-5] + '_predictions.csv'
        pred_out = os.path.join(self.path, pred_out_csv)
        prediction_status = self._set_up_prediction(pred_datafile_path,
                                                    pred_single_stats,
                                                    pred_out)
        # If pedictions already exist return file path
        if prediction_status is True:
            return pred_out
        else:
            # Start predicting if model has been trained
            try:
                features, ns_stats, s_stats = self._get_labels_and_class_stats()
                with open(pred_single_stats) as stats_csv:
                    csv_reader = csv.reader(stats_csv)
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
                            rel_ind = self._get_relevant_indices(features,
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
                        # if equal diastance classify as irony
                        if nonsarcastic_dist < sarcastic_dist:
                            prediction = 0
                        elif nonsarcastic_dist > sarcastic_dist:
                            prediction = 1
                        else:
                            prediction = 1
                        self._add_csv_entry(pred_out, [row[0],
                                                       row[1],
                                                       prediction
                                                       ]
                                            )
                logging.info(f'Prediction process completed. '
                             'Predictions in {pred_out}')
                return pred_out
            except Exception as e:
                logging.info(e)
                logging.error('Model is not trained')

    def accuracy(self, pred_csv_path):
        """Compute prediction accuracy of classified data.

        Args:
            pred_csv (str): file name of csv file containing predictions

        Return:
            accuracy (float)
        """
        if os.path.exists(pred_csv_path):
            logging.info('Computing accuracy...')
            with open(pred_csv_path) as predictions:
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
            accuracy = (correct_pred/total_pred)
            logging.info(f'Computed accuracy: {accuracy}')
            return accuracy
        else:
            logging.info('No predictions found to evaluate. '
                         'Check file name or train model and predict first.')

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

    def _set_up_prediction(self,
                           pred_data_path,
                           single_stats_path,
                           pred_out_path):
        """Set up prediction process.
        Args:
            pred_data_path (str): Path todata to predict
            single_stats_path (str): Path to csv file to save single stats
            pred_out_path (str): Path to prediction output file

        Return:
            True: If predictions already exist
            False: Otherwise
        """
        if (os.path.exists(pred_out_path) and
                os.path.getsize(pred_out_path) != 0):
            logging.info('Predictions found in {}'.format(pred_out_path))
            return True
        # If single statistics of the data to predict already
        # exists start prediction process
        # Compute single statistics otherwise
        else:
            if (os.path.exists(single_stats_path) and
                    os.path.getsize(single_stats_path) != 0):
                logging.info('Statistics for prediction already computed.')
                return False
            else:
                logging.info('Start predicting...')
                PredictionData = HeadlineData(pred_data_path)
                PredictionData.compute_single_statistics(single_stats_path)
                self._add_csv_entry(pred_out_path, ['headline',
                                                    'gold',
                                                    'prediction'
                                                    ]
                                    )
                return False

    def _get_labels_and_class_stats(self):
        """Return list of feature names of the training data and
        sarcastic and nonsarcastic statistics."""
        with open(self.class_stats) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                # first row contains feature labels of training data
                if row[0] == 'is_sarcastic':
                    all_features = row[1:]
                # is_sarcastic == 0: non sarcastic statistics
                elif int(row[0]) == 0:
                    nonsarcastic_stats = row[1:]
                # is_sarcastic == 1: sarcastic statistics
                elif int(row[0]) == 1:
                    sarcastic_stats = row[1:]
                else:
                    logging.warning('Wrong entries in class statistics')
            csv_file.close()
            return all_features, nonsarcastic_stats, sarcastic_stats

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
    parser = argparse.ArgumentParser(
        description='Process Headline Data')
    parser.add_argument('train_data',
                        help='the JSON file containing the training data')
    parser.add_argument('pred_data',
                        help='the JSON file containg the prediction data')
    args = parser.parse_args()
    start_time = time.time()
    C = Classifier(args.train_data)
    C.train_model()
    predictions = C.predict(args.pred_data)
    duration = time.time() - start_time
    logging.info(f'Running time classifier: {duration} sec')
    C.accuracy(predictions)
