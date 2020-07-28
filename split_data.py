# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Split data


import random
import logging

logging.basicConfig(filename='split_data.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


class SplitData:
    """Class for splitting data file into
    70% train, 20% test and 10% validation set
    """

    def __init__(self, data_file):
        self.data = []
        try:
            with open(data_file, 'r') as data:
                for entry in data:
                    if len(entry) > 0:
                        self.data.append(entry)
        except Exception as e:
            logging.info(e)
        finally:
            data.close()

    def train_test_val_split(self, train_file, test_file, val_file):
        logging.info(len(self.data))
        random.seed(12)
        indices = [i for i in range(len(self.data))]
        random.shuffle(indices)
        train_split_index = (len(indices) / 100) * 70
        test_split_index = (len(indices) / 100) * 90
        train_count = 0
        test_count = 0
        val_count = 0
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
