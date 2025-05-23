import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger
from src.utils.config import Config

class StockDataFetcher:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = Config(config_path)
        self.logger = get_logger(__name__, self.config.get_logging_config())
        self.data_config = self.config.get_data_config()
        
    def fetch_single(self, symbol: str, start: Optional[str] = None, 
                    end: Optional[str] = None, interval: str = '1d') -> pd.DataFrame:
        try:
            self.logger.info(f"Fetching data for {symbol}")
            
            start_date = start or self.data_config.get('start_date')
            end_date = end or self.data_config.get('end_date')
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            data['Symbol'] = symbol
            
            # Save raw data
            self._save_raw_data(data, symbol)
            
            self.logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def fetch_multiple(self, symbols: List[str], start: Optional[str] = None,
                      end: Optional[str] = None, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_single(symbol, start, end, interval)
                if not data.empty:
                    results[symbol] = data
            except Exception as e:
                self.logger.error(f"Failed to fetch {symbol}: {str(e)}")
                continue
        
        return results
    
    def fetch_latest(self, symbol: str, days: int = 30) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_single(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
    
    def _save_raw_data(self, data: pd.DataFrame, symbol: str):
        try:
            raw_data_path = os.path.join('data', 'raw', f'{symbol}.csv')
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            data.to_csv(raw_data_path)
            self.logger.info(f"Saved raw data to {raw_data_path}")
        except Exception as e:
            self.logger.error(f"Error saving raw data: {str(e)}")
    
    def load_raw_data(self, symbol: str) -> pd.DataFrame:
        raw_data_path = os.path.join('data', 'raw', f'{symbol}.csv')
        
        if not os.path.exists(raw_data_path):
            self.logger.warning(f"Raw data file not found for {symbol}")
            return pd.DataFrame()
        
        return pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
    
    def get_info(self, symbol: str) -> Dict:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            self.logger.error(f"Error getting info for {symbol}: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    fetcher = StockDataFetcher()
    
    # Fetch single stock
    data = fetcher.fetch_single('AAPL')
    print(f"Fetched {len(data)} rows for AAPL")
    
    # Fetch multiple stocks
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    all_data = fetcher.fetch_multiple(symbols)
    
    for symbol, df in all_data.items():
        print(f"{symbol}: {len(df)} rows")