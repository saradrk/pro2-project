# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Read and process data: Class for headline objects


class Headline:

    def __init__(self, headline_json, doc_obj):
        self.sarcasm = headline_json['is_sarcastic']
        self.headline = headline_json['headline']
        self.link = headline_json['article_link']
        self.doc = doc_obj
        self.features = []
        self._setup_feature_list()

    def __str__(self):
        return self.headline

    def __len__(self):
        return len(self.headline)

    def __contains__(self, token):
        return token in self.tokens

    def _setup_feature_list(self):
        self.features.extend([('headline', self.headline),
                              ('is_sarcastic', self.sarcasm)])
        self._word_based_features()
#        self._char_based_features()
#        self._sentiment_feature()

    def _sentiment_feature(self):
        # Compute average named entity count
        entities = [1 if token.ent_type > 0 else 0 for token in self.doc]
        anec = sum(entities) / len(entities)
        self.features.append(('average_named_entities', round(anec, 2)))

    def _word_based_features(self):
        # Compute average word length
        word_lengths = [len(token.text) for token in self.doc]
        awl = (sum(word_lengths) / len(word_lengths))
        self.features.append(('average_word_length', round(awl, 2)))
        # Compute average stop word count
        stop_words = [1 for token in self.doc if token.is_stop is True]
        aswc = sum(stop_words) / len(word_lengths)
        self.features.append(('average_stop_word_count', round(aswc, 2)))
        # Compute number of long words (> 5 characters)
        nlw = [wl for wl in word_lengths if wl > 6]
        self.features.append(('number_of_long_words', sum(nlw)))

    def _char_based_features(self):
        char_based_features = []
        # Compute number of quotations, don't normalize -> just one sentence
        quotations = [1 for token in self.doc if token.is_quote is True]
        num_quotes = sum(quotations) / 2
        self.features.append(('number_quotations', num_quotes))
        # Compute number of characters withour whitespace
        characters = [len(token.text) for token in self.doc]
        num_chars = sum(characters)
        self.features.append(('number_characters', num_chars))


