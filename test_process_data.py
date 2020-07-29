# Sara Derakhshani
# 29.07.2020
# Programmierung II: Projekt
# Unittests

import unittest
from process_data import HeadlineData


class HeadlineDataTestCase(unittest.TestCase):

    def setUp(self):
        self.test_data = HeadlineData('./Data/unittest_data.json')

    def test_tokens(self):
        self.assertEqual(len(self.test_data.data[0].tokens), 13)

    def test_average_word_length(self):
        test_headline = self.test_data.data[0].headline
        tokenized = test_headline.split(' ')
        wl_tokens = [len(token) for token in tokenized]
        total_l = sum(wl_tokens)
        awl = total_l/13
        test_output = next(self.test_data.average_word_lengths())
        # 4.54
        self.assertEqual(round(awl, 2), test_output[1])

if __name__ == '__main__':
    unittest.main()