# Sara Derakhshani
# 28.07.2020
# Programmierung II: Projekt
# Extract features
# Store statistics

from process_data import HeadlineData
import csv


class Classifier:

    def __init__(self, single_stats_file, class_stats_file, datafile):
        self.single_stats = single_stats_file
        self.class_stats = class_stats_file
        self.headlines = HeadlineData(datafile)

    def train_model(self):
        self.compute_single_statistics()
        sarcasm_count = 0
        nonsarcasm_count = 0
        sarcastic_stats = {'average_word_length': 0}
        nonsarcastic_stats = {'average_word_length': 0}
        with open(self.single_stats) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if int(row[2]) == 0:
                    nonsarcasm_count += 1
                    nonsarcastic_stats['average_word_length'] += float(row[3])
                else:
                    sarcasm_count += 1
                    sarcastic_stats['average_word_length'] += float(row[3])
        awl_sarcastic = (sarcastic_stats['average_word_length'] / sarcasm_count)
        awl_nonsarcastic = (nonsarcastic_stats['average_word_length'] / nonsarcasm_count)
        self.add_csv_entry(self.class_stats, [0, awl_nonsarcastic])
        self.add_csv_entry(self.class_stats, [1, awl_sarcastic])

    def compute_single_statistics(self):
        for avl in self.headlines.average_word_lengths():
            self.add_csv_entry(self.single_stats,
                               [avl[0].id,
                                avl[0].headline,
                                avl[0].sarcasm,
                                avl[1]])

    def add_csv_entry(self, csv_file, new_entry):
        """Add entry to csv file"""
        with open(csv_file, mode='a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(new_entry)
        csv_file.close()


if __name__ == '__main__':
    C = Classifier('mini_single_stats.csv',
               'mini_class_stats.csv',
               './Data/mini_data.json')
    C.train_model()