# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Read and process data: Class for headline corpus

import logging
import json
from headline import Headline
import spacy
import csv
import argparse

logging.basicConfig(filename='irony_classifier.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


class HeadlineData:
    """Class for processing and feature extraction of headline data.

    Attributes:
        data (list): list with headline objects constructed of data entries
        data_size (int): Total count of data entries
        model (str): spacy language model
        nlp (spacy Language object): spacy Language object instansiated with
            language model
        features (generator): generator of feature lists for data entries

    Methods:
        compute_single_statistics(): compute feature statistics of the data
            and save as csv file
    """

    def __init__(self, data_file, language_model='en_core_web_sm'):
        """Constructor for HeadlineData class.

        Args:
            data_file (str): name of file containing headline data
            language_model (str): spacy language model (default is
                                                        en_core_web_sm)
        """
        self.data_file = data_file
        self.data = []
        self.data_size = 0
        self.model = language_model
        self.nlp = spacy.load(language_model)
        self._process_file()
        self.features = (headline.features for headline in self.data)

    def _process_file(self):
        """Read and process data from file.

        Add Headline objects to data attribute if the data is valid,
        ignore otherwise

        Args:
            filename (str): name of file containing data
        """
        try:
            with open(self.data_file, 'r') as file:
                logging.info('Processing data...')
                entry_count = 0
                for entry in file:
                    if len(entry) > 0:
                        self.data.append(self._process_headline(entry))
                        entry_count += 1
        except Exception as e:
            logging.info(e)
        else:
            file.close()
            self.data_size += entry_count
            logging.info('{} entries processed'.format(entry_count))

    def _process_headline(self, data_entry):
        """Create and return Headline object of data entry.

        Args:
            data_entry (str): headline data entry in JSON format
        """
        json_entry = json.loads(data_entry)
        is_sarcastic = json_entry['is_sarcastic']
        headline = json_entry['headline']
        doc = self.nlp(headline)
        return Headline(is_sarcastic, headline, doc)

    def compute_single_statistics(self, out_csv):
        """Compute feature statistics for every data entry."""
        with open(out_csv, mode='w+') as out:
            writer = csv.writer(out)
            line_counter = 0
            logging.info('Computing single statistics '
                         'of {} data entries...'.format(self.data_size))
            for feature_list in self.features:
                if line_counter == 0:
                    header = [feature[0] for feature in feature_list]
                    writer.writerow(header)
                figures = [feature[1] for feature in feature_list]
                writer.writerow(figures)
                line_counter += 1
        logging.info(f'Single statistics computed and saved in {out_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process Headline Data')
    parser.add_argument('data',
                        help='the JSON file containing the headline data')
    parser.add_argument('output',
                        help='csv file to save the single feature statisics')
    args = parser.parse_args()
    HD = HeadlineData(args.data)
    HD.compute_single_statistics(args.output)
