# Sara Derakhshani
# 30.07.2020
# Programmierung II: Projekt
# Main

from classifier import Classifier
from evaluate import Evaluator
import time
import logging

logging.basicConfig(filename='process_data.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


def main():
    IronyClassifier = Classifier('Sarcasm_Headlines_Dataset_v2_train.json')
    IronyClassifier.train_model()
    IronyClassifier.predict('Sarcasm_Headlines_Dataset_v2_val.json')
    Eval = Evaluator('./csv/Val_Predictions.csv')
    # IronyClassifier = Classifier('./csv/mini_single_stats.csv',
    #                              './csv/mini_class_stats.csv',
    #                              './Data/mini_data.json')
    # IronyClassifier.train_model()
    # IronyClassifier.predict('./Data/mini_test.json',
    #                         './csv/mini_test_single_stats.csv',
    #                         './csv/mini_test_predictions.csv')
    # Eval = Evaluator('./csv/mini_test_predictions.csv')
    return Eval.accuracy()

if __name__ == '__main__':
    start_time = time.time()
    logging.info('Accuracy: {}'.format(main()))
    duration = time.time() - start_time
    logging.info(f'Running time irony classifier: {duration} sec')
