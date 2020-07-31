# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Read and process data: Class for headline corpus

import logging
import json
from headline import Headline
import spacy
import csv

logging.basicConfig(filename='process_data.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


class HeadlineData:

    def __init__(self, data_file, out_csv, language_model='en_core_web_sm'):
        self.data = []
        self.model = language_model
        self.nlp = spacy.load(language_model)
        self._process_file(data_file)
        self.features = (headline.features for headline in self.data)
        self.out_csv = out_csv

    def _process_file(self, filename):
        """Read data from file
        Add Tweet objects to tweets attribute if the data is valid,
        ignore otherwise
        """
        try:
            with open(filename, 'r') as file:
                logging.info('Processing corpus...')
                entry_count = 0
                for entry in file:
                    if len(entry) > 0:
                        self.data.append(self._process_headline(entry))
                        entry_count += 1
        except Exception as e:
            logging.info(e)
        else:
            file.close()
            logging.info('{} entries processed'.format(entry_count))

    def _process_headline(self, data_entry):
        """Create and return Headline object of data entry with ID number
        Tokenize, POS-tag, lemmatize headline string
        """
        json_entry = json.loads(data_entry)
        doc = self.nlp(json_entry['headline'])
        return Headline(json_entry, doc)

    def compute_single_statistics(self):
        with open(self.out_csv, mode='a+') as out_csv:
            writer = csv.writer(out_csv)
            line_counter = 0
            for feature_list in self.features:
                if line_counter == 0:
                    header = [feature[0] for feature in feature_list]
                    writer.writerow(header)
                figures = [feature[1] for feature in feature_list]
                writer.writerow(figures)
                line_counter += 1
        logging.info('Single statistics computed')


if __name__ == '__main__':
    HD = HeadlineData('./Data/mini_test.json', './csv/out.csv')
    HD.compute_single_statistics()
