import unittest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import DataPreprocessor
from src.data.fetch_data import StockDataFetcher

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.fetcher = StockDataFetcher()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 150, len(dates)),
            'High': np.random.uniform(100, 150, len(dates)),
            'Low': np.random.uniform(100, 150, len(dates)),
            'Close': np.random.uniform(100, 150, len(dates)),
            'Volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)
        
    def test_add_technical_indicators(self):
        # Test adding technical indicators
        data_with_indicators = self.preprocessor.add_technical_indicators(self.sample_data.copy())
        
        # Check if indicators were added
        expected_columns = ['SMA_20', 'SMA_50', 'RSI', 'Volume_SMA', 'Price_change']
        for col in expected_columns:
            self.assertIn(col, data_with_indicators.columns)
            
    def test_create_features(self):
        # Test feature creation
        features_data = self.preprocessor.create_features(self.sample_data.copy())
        
        # Check for lag features
        self.assertIn('Close_lag_1', features_data.columns)
        self.assertIn('Volume_lag_5', features_data.columns)
        
        # Check for rolling features
        self.assertIn('Close_rolling_mean_10', features_data.columns)
        self.assertIn('Volume_rolling_mean_20', features_data.columns)
        
        # Check that NaN values were dropped
        self.assertEqual(features_data.isna().sum().sum(), 0)
        
    def test_prepare_data(self):
        # Test data preparation
        X, y = self.preprocessor.prepare_data(self.sample_data.copy())
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        
        # Check that target is shifted
        self.assertLess(len(y), len(self.sample_data))
        
    def test_scale_data(self):
        # Test data scaling
        X, y = self.preprocessor.prepare_data(self.sample_data.copy())
        
        # Test fit and transform
        X_scaled, y_scaled = self.preprocessor.scale_data(X, y, fit=True)
        
        self.assertIsInstance(X_scaled, pd.DataFrame)
        self.assertEqual(X_scaled.shape, X.shape)
        
        # Check that scaling worked (mean should be close to 0, std close to 1)
        means = X_scaled.mean()
        stds = X_scaled.std()
        
        for mean in means:
            self.assertAlmostEqual(mean, 0, places=5)
        for std in stds:
            self.assertAlmostEqual(std, 1, places=1)
            
    def test_prepare_lstm_data(self):
        # Test LSTM data preparation
        X, y = self.preprocessor.prepare_data(self.sample_data.copy())
        sequence_length = 30
        
        X_lstm, y_lstm = self.preprocessor.prepare_lstm_data(X, y, sequence_length)
        
        self.assertIsInstance(X_lstm, np.ndarray)
        self.assertIsInstance(y_lstm, np.ndarray)
        
        # Check dimensions
        self.assertEqual(len(X_lstm.shape), 3)  # 3D array for LSTM
        self.assertEqual(X_lstm.shape[1], sequence_length)
        self.assertEqual(X_lstm.shape[2], X.shape[1])
        
    def test_save_and_load_processed_data(self):
        # Test saving and loading processed data
        symbol = 'TEST'
        processed_data = self.preprocessor.create_features(self.sample_data.copy())
        
        # Save
        self.preprocessor.save_processed_data(processed_data, symbol)
        
        # Check if file exists
        processed_path = os.path.join('data', 'processed', f'{symbol}_processed.csv')
        self.assertTrue(os.path.exists(processed_path))
        
        # Load
        loaded_data = self.preprocessor.load_processed_data(symbol)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), len(processed_data))

if __name__ == '__main__':
    unittest.main()