# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Read and process data: Class for headline objects


from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
# from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


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
        # self._semantic_features()
        # self._char_based_features()
        self._sentiment_feature()

    def _syntactic_features(self):
        verbs = [1 for token in self.doc if token.pos_ == 'VERB']
        n_verbs = sum(verbs)
        verb_ratio = len(verbs) / len(self._tokens())
        # self.features.append(('n_verbs', n_verbs))
        # self.features.append(('verb_ratio', verb_ratio))
        nouns = [1 for token in self.doc if token.pos_ == 'NOUN']
        n_nouns = sum(nouns)
        noun_ratio = len(nouns) / len(self._tokens())
        # self.features.append(('n_nouns', n_nouns))
        # self.features.append(('noun_ratio', noun_ratio))
        adverbs = [1 for token in self.doc if token.pos_ == 'ADV']
        n_adverbs = sum(adverbs)
        adverb_ratio = len(adverbs) / len(self._tokens())
        # self.features.append(('n_adverbs', n_adverbs))
        # self.features.append(('adverb_ratio', adverb_ratio))
        adjectives = [1 for token in self.doc if token.pos_ == 'ADJ']
        n_adjectives = sum(adjectives)
        adjective_ratio = len(adjectives) / len(self._tokens())
        self.features.append(('n_adjectives', n_adjectives))
        self.features.append(('adjective_ratio', adjective_ratio))

    def _sentiment_feature(self):
        # Compute average named entity count
        entities = [1 if token.ent_type > 0 else 0 for token in self.doc]
        anec = sum(entities) / len(entities)
 #       self.features.append(('average_named_entities', anec)) # + avl 59,699
        # Compute polarity scores
        pos_scores = []
        neg_scores = []
        adj_scores = [0]
        adv_scores = [0]
        for token in self.doc:
            pos = self._pos_for_senti_synset(token.pos_)
            if pos is not None:
                senti_input = f'{token.lemma_}.{pos}.01'
                try:
                    lemma_scores = swn.senti_synset(senti_input)
                    pos_scores.append(lemma_scores.pos_score())
                    neg_scores.append(lemma_scores.neg_score())
                    if pos == 'a':
                        adj_scores.append(lemma_scores.pos_score())
                        adj_scores.append(lemma_scores.neg_score())
                    elif pos == 'r':
                        adv_scores.append(lemma_scores.pos_score())
                        adv_scores.append(lemma_scores.neg_score())
                except Exception:
                    pos_scores.append(0)
                    neg_scores.append(0)
            else:
                pos_scores.append(0)
                neg_scores.append(0)
        # self.features.append(('pos_score_sum', sum(pos_scores))) # no 59,6
        # self.features.append(('neg_score_sum', sum(neg_scores))) # no 59,699
        self.features.append(('adj_max', max(adj_scores)))
        self.features.append(('adv_max', max(adv_scores)))
        gap = abs((sum(pos_scores) - sum(neg_scores)))
        self.features.append(('pos_neg_gap', gap))

    def _pos_for_senti_synset(self, pos_tag):
        if pos_tag == 'NOUN':
            return 'n'
        elif pos_tag == 'VERB':
            return 'v'
        elif pos_tag == 'ADJ':
            return 'a'
        elif pos_tag == 'ADV':
            return 'r'
        else:
            return None

    def _word_based_features(self):
        # Compute average word length
        word_lengths = [len(token.text) for token in self.doc]
        awl = (sum(word_lengths) / len(word_lengths))
        self.features.append(('average_word_length', awl))
        # Compute average stop word count
        stop_words = [1 for token in self.doc if token.is_stop is True]
        aswc = sum(stop_words) / len(word_lengths)
#        self.features.append(('average_stop_word_count', aswc)) # no
        # Compute number of long words (> 5 characters)
        nlw = [wl for wl in word_lengths if wl > 6]
#        self.features.append(('number_of_long_words', len(nlw))) # no

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
        self.features.append(('number_quotations', num_quotes)) # noo
        # Compute number of characters withour whitespace
        characters = [len(token.text) for token in self.doc]
        num_chars = sum(characters)
        self.features.append(('number_characters', num_chars)) # no!

