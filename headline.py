# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Read and process data: Class for headline objects


class Headline:

    def __init__(self, id, headline_json, doc_obj):
        self.sarcasm = headline_json['is_sarcastic']
        self.headline = headline_json['headline']
        self.link = headline_json['article_link']
        self.id = id
        self.tokens = [token.text for token in doc_obj]
        self.pos_tags = [token.pos_ for token in doc_obj]
        self.lemmas = [token.lemma_ for token in doc_obj]
        self.doc = doc_obj

    def __str__(self):
        return self.headline

    def __len__(self):
        return len(self.headline)

    def __contains__(self, token):
        return token in self.tokens
