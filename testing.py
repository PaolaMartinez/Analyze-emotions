import unittest
import predictor

class PredictorTest(unittest.TestCase):
    def setUp(self):
        self.predictor = predictor.Predictor("tweets_2020-04-30.csv", predictor.KNeighborsClassifier())
        self.predictor.clean_data()

    def test_data_is_cleaned(self):
        equal = all(self.predictor.data['text'] == self.predictor.processed_data)
        self.assertFalse(equal)
    
    def test_transform_normalize(self):
        data_normalized = self.predictor.transform_normalize(self.predictor.processed_data)
        self.assertNotEqual(self.predictor.processed_data,data_normalized)
    
    def test_split(self):
        data_normalized = self.predictor.transform_normalize(self.predictor.processed_data)
        self.predictor.split(data_normalized,0.2)
        if hasattr(self.predictor, 'X_train') and hasattr(self.predictor, 'X_test') and hasattr(self.predictor, 'y_train') and hasattr(self.predictor, 'y_test'):
            correct = True
        else:
            correct = False
        self.assertTrue(correct)
    
    def test_train_model(self):
        data_normalized = self.predictor.transform_normalize(self.predictor.processed_data)
        classifier = self.predictor.train_model(data_normalized, 0.2)
        result = classifier.score(self.predictor.X_test, self.predictor.y_test)
        self.assertEqual(result, 0.52)

if __name__ == "__main__":
    unittest.main()