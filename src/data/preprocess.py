import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ta
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger
from src.utils.config import Config

class DataPreprocessor:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = Config(config_path)
        self.logger = get_logger(__name__, self.config.get_logging_config())
        self.data_config = self.config.get_data_config()
        self.training_config = self.config.get_training_config()
        self.scaler = None
        
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Adding technical indicators")
        
        # Simple Moving Averages
        if 'SMA_20' in self.data_config.get('features', []):
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        
        if 'SMA_50' in self.data_config.get('features', []):
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        
        # RSI
        if 'RSI' in self.data_config.get('features', []):
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        
        # MACD
        if 'MACD' in self.data_config.get('features', []):
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        if 'BB_upper' in self.data_config.get('features', []) or 'BB_lower' in self.data_config.get('features', []):
            bb = ta.volatility.BollingerBands(data['Close'])
            data['BB_upper'] = bb.bollinger_hband()
            data['BB_lower'] = bb.bollinger_lband()
            data['BB_middle'] = bb.bollinger_mavg()
        
        # ATR (Average True Range)
        if 'ATR' in self.data_config.get('features', []):
            data['ATR'] = ta.volatility.average_true_range(
                data['High'], data['Low'], data['Close']
            )
        
        # Volume indicators
        data['Volume_SMA'] = ta.trend.sma_indicator(data['Volume'], window=20)
        data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Price features
        data['Price_change'] = data['Close'].pct_change()
        data['High_Low_ratio'] = (data['High'] - data['Low']) / data['Close']
        data['Close_Open_ratio'] = (data['Close'] - data['Open']) / data['Open']
        
        return data
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Creating features")
        
        # Add technical indicators
        data = self.add_technical_indicators(data)
        
        # Create lag features
        for lag in [1, 2, 3, 5, 10]:
            data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_lag_{lag}'] = data['Volume'].shift(lag)
        
        # Create rolling statistics
        for window in [5, 10, 20]:
            data[f'Close_rolling_mean_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'Close_rolling_std_{window}'] = data['Close'].rolling(window=window).std()
            data[f'Volume_rolling_mean_{window}'] = data['Volume'].rolling(window=window).mean()
        
        # Drop rows with NaN values
        data = data.dropna()
        
        return data
    
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'Close',
                    predict_days: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        self.logger.info("Preparing data for training")
        
        # Create features
        data = self.create_features(data)
        
        # Create target variable (future price)
        data['Target'] = data[target_col].shift(-predict_days)
        
        # Drop last rows without target
        data = data.dropna()
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col not in ['Target', 'Symbol']]
        X = data[feature_cols]
        y = data['Target']
        
        return X, y
    
    def scale_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                  fit: bool = True) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        scaling_method = self.training_config.get('scaling', 'standard')
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        if fit:
            self.scaler = scaler
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        if y is not None:
            return X_scaled, y
        return X_scaled
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names after preprocessing"""
        # Create a dummy DataFrame to get feature names
        dummy_data = pd.DataFrame({
            'Open': [100], 'High': [101], 'Low': [99], 
            'Close': [100], 'Volume': [1000000]
        })
        dummy_features = self.create_features(dummy_data)
        return [col for col in dummy_features.columns if col not in ['Symbol']]
    
    def save_processed_data(self, data: pd.DataFrame, symbol: str):
        try:
            processed_path = os.path.join('data', 'processed', f'{symbol}_processed.csv')
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            data.to_csv(processed_path)
            self.logger.info(f"Saved processed data to {processed_path}")
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
    
    def load_processed_data(self, symbol: str) -> pd.DataFrame:
        processed_path = os.path.join('data', 'processed', f'{symbol}_processed.csv')
        
        if not os.path.exists(processed_path):
            self.logger.warning(f"Processed data file not found for {symbol}")
            return pd.DataFrame()
        
        return pd.read_csv(processed_path, index_col=0, parse_dates=True)

if __name__ == "__main__":
    # Example usage
    from src.data.fetch_data import StockDataFetcher
    
    # Fetch data
    fetcher = StockDataFetcher()
    data = fetcher.fetch_single('AAPL')
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(data)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {X.columns.tolist()}")