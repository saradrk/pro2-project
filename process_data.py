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

    def __init__(self, data_file, language_model):
        self.data = []
        self.model = language_model
        self.nlp = spacy.load(language_model)
        self._process_file(data_file)

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
        json_entry = json.loads(data_entry)
        doc = self.nlp(json_entry['headline'])
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        lemmas = [token.lemma_ for token in doc]
        return Headline(json_entry, tokens, pos_tags, lemmas)
        


if __name__ == '__main__':
    HD = HeadlineData('./Data/mini_data.json', 'en_core_web_sm')
