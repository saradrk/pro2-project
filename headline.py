# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Read and process data: Class for headline objects

from nltk.corpus import sentiwordnet as swn


class Headline:
    """Class for computing linguistic features of a newspaper headline.

    Attributes:
        is_sarcastic (int): 1 if sarcastic 0 otherwise
        headline (str): the newspaper headline
        link (str): article link
        doc (object of spacy Doc class): Doc object instantiated with headline
        features (list): list of linguistic features

    No public methods
    """

    def __init__(self, is_sarcastic, headline, doc_obj, link=None):
        """Constructor for Headline class.

        Args:
            is_sarcastic (int): 1 if sarcastic 0 otherwise
            headline (str): the newspaper headline
            doc_abj (spacy Doc object): Doc object instantiated with headline
            link (str): article link (optional)
        """
        self.is_sarcastic = is_sarcastic
        self.headline = headline
        self.link = link
        self.doc = doc_obj
        self.features = []
        self._setup_feature_list()

    def __str__(self):
        """Return headline as string."""
        return self.headline

    def __len__(self):
        """Return number of characters in headline."""
        return len(self.headline)

    def __contains__(self, token):
        """Return True if headline contains token, False otherwise.

        Args:
            token (str): word that should be checked for occurence in headline
        """
        return token in self._tokens()

    def _tokens(self):
        """Return list of tokens."""
        tokens = [token.text for token in self.doc]
        return tokens

    def _lemmas(self):
        """Return list of lemmas."""
        lemmas = [token.lemma_ for token in self.doc]
        return lemmas

    def _setup_feature_list(self):
        """Call feature methods to fill features attribute list."""
        self.features.extend([('headline', self.headline),
                              ('is_sarcastic', self.is_sarcastic)])
        self._extract_word_based_features()
        self._extract_syntactic_features()
        self._extract_sentiment_feature()

    def _extract_word_based_features(self):
        """Add word based features to features attribute."""
        # Compute and add average word length
        word_lengths = [len(token.text) for token in self.doc]
        awl = (sum(word_lengths) / len(word_lengths))
        self.features.append(('average_word_length', awl))

    def _extract_syntactic_features(self):
        """Add features of syntactic nature to features attribute."""
        adjectives = [1 for token in self.doc if token.pos_ == 'ADJ']
        # total number of adjectives
        n_adjectives = sum(adjectives)
        # proportion of adjectives
        adjective_ratio = len(adjectives) / len(self._tokens())
        self.features.append(('n_adjectives', n_adjectives))
        self.features.append(('adjective_ratio', adjective_ratio))

    def _extract_sentiment_feature(self):
        """Add sentiment features to features attribute."""
        # Compute polarity scores
        pos_scores = []  # positive polarity scores
        neg_scores = []  # negative polarity scores
        adj_scores = [0]  # polarity scores of adjectives
        adv_scores = [0]  # polarity scores of adverbs
        for token in self.doc:
            # get POS tag used by sentiwordnet
            pos = self._pos_for_senti_synset(token.pos_)
            # if matching POS tag exists get polarity score with lemma of the
            # token, POS tag and 01 (calls for most common usage of the word)
            # append 0 to score lists otherwise
            if pos is not None:
                senti_input = f'{token.lemma_}.{pos}.01'
                # add positive and negative scores to lists if entry exists
                # append 0 otherwise
                try:
                    lemma_scores = swn.senti_synset(senti_input)
                    pos_scores.append(lemma_scores.pos_score())
                    neg_scores.append(lemma_scores.neg_score())
                    # if token is an adjective add scores to list
                    if pos == 'a':
                        adj_scores.append(lemma_scores.pos_score())
                        adj_scores.append(lemma_scores.neg_score())
                    # if token is an adverb add scores to list
                    elif pos == 'r':
                        adv_scores.append(lemma_scores.pos_score())
                        adv_scores.append(lemma_scores.neg_score())
                except Exception:
                    pos_scores.append(0)
                    neg_scores.append(0)
            else:
                pos_scores.append(0)
                neg_scores.append(0)
        # Add highest polarity score of adjectives
        self.features.append(('adj_max', max(adj_scores)))
        # Add highest polarity score of adverbs
        self.features.append(('adv_max', max(adv_scores)))
        # gap between positive and negative scores
        gap = abs((sum(pos_scores) - sum(neg_scores)))
        self.features.append(('pos_neg_gap', gap))

    def _pos_for_senti_synset(self, pos_tag):
        """Return matching sentiwordnet POS tag to spacy POS tag.

        Args:
            pos_tag (str): POS tag from spacy
        """
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
