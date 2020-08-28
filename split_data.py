# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Split data


import random
import logging
import os

logging.basicConfig(filename='irony_classifier.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


class SplitData:
    """Class for splitting a data file into training/testing/validation set.

    Attributes:
        data_file (str): name of data file
        data (list): list of data entries

    Methods:
        train_test_val_split(train_file, test_file, val_file): split data
    """

    def __init__(self, data_file):
        """Constructor for SplitData class

        Args:
            data_file (str): name of file containing the data
        """
        self.data_file = os.path.join('Data', data_file)
        self.data = []
        try:
            self._read_data()
        except Exception as e:
            logging.critical(e)

    def _read_data(self):
        with open(self.data_file, 'r') as data:
            for entry in data:
                if len(entry) > 0:
                    self.data.append(entry)
        self.data_file.close()

    def train_test_val_split(self,
                             train_filename,
                             test_filename,
                             val_filename):
        """Split data into training/test/validation set (70%/20%/10%).

        Save split sets as seperate JSON files.

        Args:
            train_filename (str): file name for training data
            test_filename (str): file name for test data
            val_filename (str): file name for validation data
        """
        train_file = os.path.join('Data', train_filename)
        test_file = os.path.join('Data', test_filename)
        val_file = os.path.join('Data', val_filename)
        logging.info(len(self.data))
        random.seed(12)
        # Create list with randomly shuffled indices for
        # randomly distributing the data
        indices = [i for i in range(len(self.data))]
        random.shuffle(indices)
        train_split_index = (len(indices) / 100) * 70
        test_split_index = (len(indices) / 100) * 90
        # Count entries for info log
        train_count = 0
        test_count = 0
        val_count = 0
        # If files haven't been splitted yet:
        # Add random 70% of data entries to training file using first 70% of
        # random indices, use next 20% of random indices for test and
        # remaining approximately 10% for validation file
        if open(train_file, 'r'):
            logging.info('Corpus has already been split')
        else:
            for i in range(len(self.data)):
                if i <= train_split_index:
                    with open(train_file, 'a+') as train_out:
                        train_out.write(self.data[indices[i]])
                    train_count += 1
                elif i <= test_split_index:
                    with open(test_file, 'a+') as test_out:
                        test_out.write(self.data[indices[i]])
                    test_count += 1
                else:
                    with open(val_file, 'a+') as val_out:
                        val_out.write(self.data[indices[i]])
                    val_count += 1
            logging.info('Created training set of size {}'.format(train_count))
            logging.info('Created test set of size {}'.format(test_count))
            logging.info('Created validation set of size {}'.format(val_count))


if __name__ == '__main__':
    Data = SplitData('./Data/Sarcasm_Headlines_Dataset_v2.json')
    Data.train_test_val_split('./Data/Sarcasm_Headlines_Dataset_v2_train.json',
                              './Data/Sarcasm_Headlines_Dataset_v2_test.json',
                              './Data/Sarcasm_Headlines_Dataset_v2_val.json')
