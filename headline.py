# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Read and process data: Class for headline objects


class Headline:

    def __init__(self, id, headline_json, tokens, pos_tags, lemmas):
        self.sarcasm = headline_json['is_sarcastic']
        self.headline = headline_json['headline']
        self.link = headline_json['article_link']
        self.id = id
        self.tokens = tokens
        self.pos_tags = pos_tags
        self.lemmas = lemmas

    def __str__(self):
        return self.headline

    def __len__(self):
        return len(self.headline)

    def __contains__(self, token):
        return token in self.tokens
