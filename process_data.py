# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Read and process data: Class for headline corpus

import logging
import json
from headline import Headline
import spacy

logging.basicConfig(filename='process_data.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


class HeadlineData:

    def __init__(self, data_file, language_model='en_core_web_sm'):
        self.data = []
        self.model = language_model
        self.nlp = spacy.load(language_model)
        self._process_file(data_file)
        self.features = self._compute_features()

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
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        lemmas = [token.lemma_ for token in doc]
        return Headline(id_number, json_entry, tokens, pos_tags, lemmas)

    def average_word_lengths(self):
        """Generator to get average word length"""
        n = 0
        while n < len(self.data):
            # lengths of tokens in headline
            word_lengths = [len(token) for token in self.data[n].tokens]
            # average length of tokens
            average_word_length = (sum(word_lengths) / len(word_lengths))
            yield (self.data[n], round(average_word_length, 2))
            n += 1

    def _compute_features(self):
        return [self.average_word_lengths()]


if __name__ == '__main__':
    HD = HeadlineData('./Data/mini_data.json')
    
