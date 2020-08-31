# Sara Derakhshani
# 30.07.2020
# Programmierung II: Projekt
# Main

from classifier import Classifier
import time
import logging
import os

logging.basicConfig(filename='irony_classifier.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


def main():
    train_data_path = os.path.join('Data',
                                   'Sarcasm_Headlines_Dataset_v2_train.json')
    pred_data_path = os.path.join('Data',
                                  'Sarcasm_Headlines_Dataset_v2_test.json')
    IronyClassifier = Classifier(train_data_path)
    IronyClassifier.train_model()
    pred_csv = IronyClassifier.predict(pred_data_path)
    return IronyClassifier.accuracy(pred_csv)


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = time.time() - start_time
    logging.info(f'Running time irony classifier: {duration} sec')
    print('Finished classification process.')
