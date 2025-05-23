import os
import sys
import argparse
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger
from src.utils.config import Config
from src.data.fetch_data import StockDataFetcher
from src.data.preprocess import DataPreprocessor

class ModelTrainer:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = Config(config_path)
        self.logger = get_logger(__name__, self.config.get_logging_config())
        self.model_config = self.config.get_model_config()
        self.training_config = self.config.get_training_config()
        self.model = None
        self.preprocessor = DataPreprocessor(config_path)
        
    def prepare_training_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.Series]:
        # Fetch data
        fetcher = StockDataFetcher(self.config.config_path)
        data = fetcher.fetch_single(symbol)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Prepare features and target
        X, y = self.preprocessor.prepare_data(data)
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        test_size = self.training_config.get('test_size', 0.2)
        val_size = self.training_config.get('validation_size', 0.1)
        random_state = self.training_config.get('random_state', 42)
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=False
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> RandomForestRegressor:
        self.logger.info("Training Random Forest model")
        
        params = self.model_config['hyperparameters']['random_forest']
        
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 15),
            min_samples_split=params.get('min_samples_split', 5),
            min_samples_leaf=params.get('min_samples_leaf', 2),
            random_state=self.training_config.get('random_state', 42),
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Validate
        val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_pred)
        self.logger.info(f"Validation MSE: {val_mse:.4f}")
        
        return model
    
    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> GradientBoostingRegressor:
        self.logger.info("Training Gradient Boosting model")
        
        params = self.model_config['hyperparameters'].get('gradient_boosting', {})
        
        model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.8),
            random_state=self.training_config.get('random_state', 42)
        )
        
        model.fit(X_train, y_train)
        
        # Validate
        val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_pred)
        self.logger.info(f"Validation MSE: {val_mse:.4f}")
        
        return model
    
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> LinearRegression:
        self.logger.info("Training Linear Regression model")
        
        params = self.model_config['hyperparameters'].get('linear_regression', {})
        
        model = LinearRegression(
            fit_intercept=params.get('fit_intercept', True)
        )
        
        model.fit(X_train, y_train)
        
        # Validate
        val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_pred)
        self.logger.info(f"Validation MSE: {val_mse:.4f}")
        
        return model
    
    def train(self, symbol: str, algorithm: Optional[str] = None) -> Any:
        algorithm = algorithm or self.model_config.get('algorithm', 'gradient_boosting')
        self.logger.info(f"Training {algorithm} model for {symbol}")
        
        # Prepare data
        X, y = self.prepare_training_data(symbol)
        
        # Scale data
        X_scaled, y = self.preprocessor.scale_data(X, y, fit=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_scaled, y)
        
        # Train model based on algorithm
        if algorithm == 'random_forest':
            self.model = self.train_random_forest(X_train, y_train, X_val, y_val)
        elif algorithm == 'gradient_boosting':
            self.model = self.train_gradient_boosting(X_train, y_train, X_val, y_val)
        elif algorithm == 'linear_regression':
            self.model = self.train_linear_regression(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Evaluate on test set
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred),
            'r2': r2_score(y_test, test_pred)
        }
        
        self.logger.info(f"Test metrics: {metrics}")
        
        # Save model
        self.save_model(symbol, algorithm, metrics)
        
        return self.model
    
    def save_model(self, symbol: str, algorithm: str, metrics: Dict[str, float]):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{symbol}_{algorithm}_{timestamp}"
        model_path = os.path.join('models', 'trained', f'{model_name}.pkl')
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        joblib.dump(self.preprocessor.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'algorithm': algorithm,
            'timestamp': timestamp,
            'metrics': metrics,
            'config': {
                'model': self.model_config,
                'training': self.training_config
            }
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        self.logger.info(f"Model saved to {model_path}")
        
        # Create symlink to latest model
        latest_path = os.path.join('models', 'trained', f'{symbol}_{algorithm}_latest.pkl')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(model_path), latest_path)

def main():
    parser = argparse.ArgumentParser(description='Train stock prediction model')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--algorithm', type=str, default='gradient_boosting',
                       choices=['random_forest', 'gradient_boosting', 'linear_regression'])
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.config)
    model = trainer.train(args.symbol, args.algorithm)
    print(f"Model training completed for {args.symbol}")

if __name__ == "__main__":
    main()