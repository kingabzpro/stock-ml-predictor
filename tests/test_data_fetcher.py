import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fetch_data import StockDataFetcher

class TestStockDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = StockDataFetcher()
        
    def test_fetch_single_stock(self):
        # Test fetching data for a single stock
        symbol = 'AAPL'
        data = self.fetcher.fetch_single(symbol, start='2023-01-01', end='2023-01-31')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn('Symbol', data.columns)
        self.assertEqual(data['Symbol'].iloc[0], symbol)
        
    def test_fetch_multiple_stocks(self):
        # Test fetching data for multiple stocks
        symbols = ['AAPL', 'GOOGL']
        data_dict = self.fetcher.fetch_multiple(symbols, start='2023-01-01', end='2023-01-31')
        
        self.assertIsInstance(data_dict, dict)
        self.assertEqual(len(data_dict), 2)
        
        for symbol in symbols:
            self.assertIn(symbol, data_dict)
            self.assertIsInstance(data_dict[symbol], pd.DataFrame)
            
    def test_fetch_latest_data(self):
        # Test fetching latest data
        symbol = 'MSFT'
        data = self.fetcher.fetch_latest(symbol, days=10)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        
    def test_invalid_symbol(self):
        # Test with invalid symbol
        symbol = 'INVALID_SYMBOL_XYZ'
        data = self.fetcher.fetch_single(symbol, start='2023-01-01', end='2023-01-31')
        
        # Should return empty DataFrame for invalid symbols
        self.assertTrue(data.empty)
        
    def test_save_and_load_raw_data(self):
        # Test saving and loading raw data
        symbol = 'AAPL'
        data = self.fetcher.fetch_single(symbol, start='2023-01-01', end='2023-01-31')
        
        # Check if file was saved
        raw_data_path = os.path.join('data', 'raw', f'{symbol}.csv')
        self.assertTrue(os.path.exists(raw_data_path))
        
        # Test loading
        loaded_data = self.fetcher.load_raw_data(symbol)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertFalse(loaded_data.empty)

if __name__ == '__main__':
    unittest.main()