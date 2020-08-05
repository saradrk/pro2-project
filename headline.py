# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Read and process data: Class for headline objects
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


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
        return token in self._tokens()

    def _tokens(self):
        tokens = [token.text for token in self.doc]
        return tokens

    def _lemmas(self):
        lemmas = [token.lemma_ for token in self.doc]
        return lemmas

    def _setup_feature_list(self):
        self.features.extend([('headline', self.headline),
                              ('is_sarcastic', self.sarcasm)])
        self._word_based_features()
        self._syntactic_features()
        self._semantic_features()
#        self._char_based_features()
#        self._sentiment_feature()

    def _syntactic_features(self):
        verbs = [1 for token in self.doc if token.pos_ == 'VERB']
        n_verbs = sum(verbs)
        verb_ratio = len(verbs) / len(self._tokens())
#        self.features.append(('n_verbs', n_verbs))
        self.features.append(('verb_ratio', verb_ratio))
        nouns = [1 for token in self.doc if token.pos_ == 'NOUN']
        n_nouns = sum(nouns)
        noun_ratio = len(nouns) / len(self._tokens())
#        self.features.append(('n_nouns', n_nouns))
        self.features.append(('noun_ratio', noun_ratio))
        adverbs = [1 for token in self.doc if token.pos_ == 'ADV']
        n_adverbs = sum(adverbs)
        adverb_ratio = len(adverbs) / len(self._tokens())
#        self.features.append(('n_adverbs', n_adverbs))
        self.features.append(('adverb_ratio', adverb_ratio))
        adjectives = [1 for token in self.doc if token.pos_ == 'ADJ']
        n_adjectives = sum(adjectives)
        adjective_ratio = len(adjectives) / len(self._tokens())
#        self.features.append(('n_adjectives', n_adjectives))
        self.features.append(('adjective_ratio', adjective_ratio))

    def _sentiment_feature(self):
        # Compute average named entity count
        entities = [1 if token.ent_type > 0 else 0 for token in self.doc]
        anec = sum(entities) / len(entities)
        self.features.append(('average_named_entities', anec))

    def _word_based_features(self):
        # Compute average word length
        word_lengths = [len(token.text) for token in self.doc]
        awl = (sum(word_lengths) / len(word_lengths))
        self.features.append(('average_word_length', awl))
        # Compute average stop word count
        stop_words = [1 for token in self.doc if token.is_stop is True]
        aswc = sum(stop_words) / len(word_lengths)
        self.features.append(('average_stop_word_count', aswc))
        # Compute number of long words (> 5 characters)
        nlw = [wl for wl in word_lengths if wl > 6]
        self.features.append(('number_of_long_words', sum(nlw)))

    def _semantic_features(self):
        n_syn_per_token = []
        for lemma in self._lemmas():
            synonyms = 0
            for syn in wn.synsets(lemma):
                synonyms += 1
            n_syn_per_token.append(synonyms)
        max_synset = max(n_syn_per_token)
        synset_mean = sum(n_syn_per_token) / len(self._lemmas())
        self.features.append(('max_synset', max_synset))
        self.features.append(('synset_mean', synset_mean))

    def _char_based_features(self):
        # Compute number of quotations, don't normalize -> just one sentence
        quotations = [1 for token in self.doc if token.is_quote is True]
        num_quotes = sum(quotations) / 2
        self.features.append(('number_quotations', num_quotes))
        # Compute number of characters withour whitespace
        characters = [len(token.text) for token in self.doc]
        num_chars = sum(characters)
        self.features.append(('number_characters', num_chars))


