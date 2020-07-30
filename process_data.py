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
        self.features = self._generate_features()
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
                        self.data.append(self._process_headline(entry,
                                                                entry_count))
                        entry_count += 1
        except Exception as e:
            logging.info(e)
        else:
            file.close()
            logging.info('{} entries processed'.format(entry_count))

    def _process_headline(self, data_entry, id_number):
        """Create and return Headline object of data entry with ID number
        Tokenize, POS-tag, lemmatize headline string
        """
        json_entry = json.loads(data_entry)
        doc = self.nlp(json_entry['headline'])
        return Headline(id_number, json_entry, doc)

    def _generate_features(self):
        n = 0
        while n < len(self.data):
            awl = self._awl(self.data[n])
            aswc = self._average_stop_word_count(self.data[n])
            yield [self.data[n].id,
                   self.data[n].headline,
                   self.data[n].sarcasm,
                   awl,
                   aswc]
            n += 1

    def _awl(self, headline):
        word_lengths = [len(token) for token in headline.tokens]
        average_word_length = (sum(word_lengths) / len(word_lengths))
        return round(average_word_length, 2)

    def _average_stop_word_count(self, headline):
        stop_words = [1 for token in headline.doc if token.is_stop is True]
        aswc = sum(stop_words) / len(headline.tokens)
        return round(aswc, 2)

    def compute_single_statistics(self):
        with open(self.out_csv, mode='a+') as out_csv:
            writer = csv.writer(out_csv)
            for feature_list in self.features:
                writer.writerow(feature_list)
        logging.info('Single statistics computed')


    # def average_word_lengths(self):
    #     """Generator to get average word length"""
    #     n = 0
    #     while n < len(self.data):
    #         # lengths of tokens in headline
    #         word_lengths = [len(token) for token in self.data[n].tokens]
    #         # average length of tokens
    #         average_word_length = (sum(word_lengths) / len(word_lengths))
    #         yield (self.data[n], round(average_word_length, 2))
    #         n += 1

    # def stop_word_count(self):
    #     n = 0
    #     while n < len(self.data)

    # def _compute_features(self):
    #     return [self.average_word_lengths()]


if __name__ == '__main__':
    HD = HeadlineData('./Data/mini_data.json')
    
