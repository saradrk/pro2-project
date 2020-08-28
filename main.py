# Sara Derakhshani
# 30.07.2020
# Programmierung II: Projekt
# Main

from classifier import Classifier
import time
import logging

logging.basicConfig(filename='process_data.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


def main():
    IronyClassifier = Classifier('Sarcasm_Headlines_Dataset_v2_train.json')
    IronyClassifier.train_model()
    IronyClassifier.predict('Sarcasm_Headlines_Dataset_v2_val.json')
    return IronyClassifier.accuracy()


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = time.time() - start_time
    logging.info(f'Running time irony classifier: {duration} sec')
