import unittest
import pandas as pd
from bibmon._outlier_handling import detect_outliers_iqr, remove_outliers

class TestOutlierHandling(unittest.TestCase):

    def test_detect_outliers_iqr(self):
        # Create a sample DataFrame with outliers
        data = {'col1': [1, 2, 3, 4, 5, 100]}
        df = pd.DataFrame(data)

        # Run the detect_outliers_iqr function
        df_outliers = detect_outliers_iqr(df, ['col1'])

        # Check if the outlier was detected correctly
        self.assertEqual(df_outliers['col1'].tolist(), [0, 0, 0, 0, 0, 1])

    def test_remove_outliers_remove(self):
        # Create a sample DataFrame with outliers
        data = {'col1': [1, 2, 3, 4, 5, 100]}
        df = pd.DataFrame(data)

        # Run the remove_outliers function with method='remove'
        df_outliers = remove_outliers(df, ['col1'], method='remove')

        # Check if the outlier was removed correctly
        self.assertEqual(df_outliers['col1'].tolist(), [1, 2, 3, 4, 5])

    def test_remove_outliers_median(self):
        # Create a sample DataFrame with outliers
        data = {'col1': [1, 2, 3, 4, 5, 100]}
        df = pd.DataFrame(data)

        # Run the remove_outliers function with method='median'
        df_outliers = remove_outliers(df, ['col1'], method='median')

        # Check if the outlier was replaced by the median
        self.assertEqual(df_outliers['col1'].tolist(), [1, 2, 3, 4, 5, 3])

    def test_remove_outliers_winsorize(self):
        # Create a sample DataFrame with outliers
        data = {'col1': [1, 2, 3, 4, 5, 100]}
        df = pd.DataFrame(data)

        # Run the remove_outliers function with method='winsorize'
        df_outliers = remove_outliers(df, ['col1'], method='winsorize')

        # Check if the outlier was winsorized
        self.assertTrue(df_outliers['col1'].tolist()[-1] < 100)  # Check if the value was limited

if __name__ == '__main__':
    unittest.main()