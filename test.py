# Sara Derakhshani
# 29.07.2020
# Programmierung II: Projekt
# Unittests

import unittest
from process_data import HeadlineData
from classifier import Classifier
import os
import csv


class HeadlineDataTestCase(unittest.TestCase):
  
  def setUp(self):
    datafile = os.path.join('test_data', 'test_data.json')
    self.TestData = HeadlineData(datafile)
    
    def test_process_file(self):
      self.assertEqual(len(self.TestData.data), 2)
    
    def test_process_headline(self):
      with open(self.TestData.data_file, 'r') as data:
        for entry in data:
          processed_headline = self.TestData._process_headline(entry)
          self.assertEqual(type(processed_headline).__name__,
                           'Headline')
    
    def test_compute_single_statistics(self):
      test_single_stats = os.path.join('test_data', 'test_single_stats.csv')
      self.TestData.compute_single_statistics(test_single_stats)
      with open(test_single_stats) as stats:
        reader = csv.reader(stats)
        csv_labels = next(reader)
        self.assertEqual(len(csv_labels), 8)


class ClassifierTestCase(unittest.TestCase):
  
  def setUp(self):
    datafile = os.path.join('test_data', 'test_data.json')
    self.pred_datafile = os.path.join('test_data',
                                      'pred_test_data.json')
                                      self.Classifier = Classifier(datafile)
                                        self.Classifier.train_model()
                                      
                                      def test_train_model(self):
                                        with open(self.Classifier.class_stats) as stats:
                                          class_reader = csv.reader(stats)
                                          next(class_reader)
                                          first_row_class_stats = next(class_reader)
                                            with open(self.Classifier.single_stats) as stats:
                                              single_reader = csv.reader(stats)
                                              next(single_reader)
                                              first_row_single_stats = next(single_reader)
                                                self.assertEqual(first_row_class_stats, first_row_single_stats[1:])
                                                  
                                                  def test_predict(self):
                                                    pred_csv = self.Classifier.predict(self.pred_datafile)
                                                    with open(pred_csv) as pred_csv:
                                                      pred_reader = csv.reader(pred_csv)
                                                      next(pred_reader)
                                                      first_row_predictions = next(pred_reader)
                                                        self.assertEqual(first_row_predictions[2], '1')
                                                          
                                                          def test_accuracy(self):
                                                            pred_csv = self.Classifier.predict(self.pred_datafile)
                                                            accuracy = self.Classifier.accuracy(pred_csv)
                                                            self.assertEqual(accuracy, 1)


if __name__ == '__main__':
  unittest.main()
