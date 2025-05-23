import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger
from src.utils.config import Config
from src.data.fetch_data import StockDataFetcher
from src.data.preprocess import DataPreprocessor

class StockPredictor:
    def __init__(self, model_path: str, config_path: str = 'config.yaml'):
        self.config = Config(config_path)
        self.logger = get_logger(__name__, self.config.get_logging_config())
        self.model_path = model_path
        self.model = self._load_model()
        self.metadata = self._load_metadata()
        self.scaler = self._load_scaler()
        self.preprocessor = DataPreprocessor(config_path)
        # Important: Set the loaded scaler to the preprocessor
        if self.scaler is not None:
            self.preprocessor.scaler = self.scaler
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        return joblib.load(self.model_path)
    
    def _load_metadata(self):
        metadata_path = self.model_path.replace('.pkl', '_metadata.pkl')
        if os.path.exists(metadata_path):
            return joblib.load(metadata_path)
        return {}
    
    def _load_scaler(self):
        scaler_path = self.model_path.replace('.pkl', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.logger.info(f"Loading scaler from {scaler_path}")
            return joblib.load(scaler_path)
        self.logger.warning("No scaler found for the model")
        return None
    
    def predict_next_day(self, symbol: str, current_data: Optional[pd.DataFrame] = None) -> float:
        self.logger.info(f"Predicting next day price for {symbol}")
        
        # Fetch recent data if not provided
        if current_data is None:
            fetcher = StockDataFetcher(self.config.config_path)
            current_data = fetcher.fetch_latest(symbol, days=100)
        
        # Prepare features
        X, _ = self.preprocessor.prepare_data(current_data)
        
        # Take the last row for prediction
        X_last = X.iloc[[-1]]
        
        # Scale features using the loaded scaler
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_last),
                columns=X_last.columns,
                index=X_last.index
            )
        else:
            self.logger.warning("No scaler available, using unscaled features")
            X_scaled = X_last
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        return float(prediction)
    
    def predict_multiple_days(self, symbol: str, days: int = 5,
                            current_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.logger.info(f"Predicting {days} days for {symbol}")
        
        # Fetch recent data if not provided
        if current_data is None:
            fetcher = StockDataFetcher(self.config.config_path)
            current_data = fetcher.fetch_latest(symbol, days=100)
        
        predictions = []
        dates = []
        
        # Start from the last date in the data
        last_date = current_data.index[-1]
        
        for i in range(days):
            # Predict next day
            next_price = self.predict_next_day(symbol, current_data)
            
            # Calculate next business day
            next_date = last_date + timedelta(days=1)
            while next_date.weekday() in [5, 6]:  # Skip weekends
                next_date += timedelta(days=1)
            
            predictions.append(next_price)
            dates.append(next_date)
            
            # Update current_data with prediction for recursive prediction
            # This is a simplified approach - in practice, you'd need to update all features
            new_row = pd.DataFrame({
                'Open': [next_price],
                'High': [next_price * 1.01],  # Simplified
                'Low': [next_price * 0.99],   # Simplified
                'Close': [next_price],
                'Volume': [current_data['Volume'].mean()]
            }, index=[next_date])
            
            current_data = pd.concat([current_data, new_row])
            last_date = next_date
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Price': predictions
        })
        predictions_df.set_index('Date', inplace=True)
        
        return predictions_df
    
    def predict_batch(self, features_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        self.logger.info("Making batch predictions")
        
        predictions = {}
        
        for symbol, features in features_dict.items():
            try:
                # Convert features to DataFrame
                X = pd.DataFrame([features])
                
                # Scale features using the loaded scaler
                if self.scaler is not None:
                    X_scaled = pd.DataFrame(
                        self.scaler.transform(X),
                        columns=X.columns,
                        index=X.index
                    )
                else:
                    self.logger.warning("No scaler available, using unscaled features")
                    X_scaled = X
                
                # Make prediction
                prediction = self.model.predict(X_scaled)[0]
                predictions[symbol] = float(prediction)
                
            except Exception as e:
                self.logger.error(f"Error predicting for {symbol}: {str(e)}")
                predictions[symbol] = None
        
        return predictions
    
    def predict_with_confidence(self, symbol: str, 
                              current_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        self.logger.info(f"Predicting with confidence intervals for {symbol}")
        
        # This is a simplified confidence calculation
        # For proper confidence intervals, you'd need:
        # - Ensemble models for uncertainty estimation
        # - Bayesian approaches
        # - Quantile regression
        
        prediction = self.predict_next_day(symbol, current_data)
        
        # Simple confidence based on historical volatility
        if current_data is not None:
            returns = current_data['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # 95% confidence interval (approximately 2 standard deviations)
            confidence_interval = 1.96 * volatility * prediction
            
            return {
                'prediction': prediction,
                'lower_bound': prediction - confidence_interval,
                'upper_bound': prediction + confidence_interval,
                'confidence': 0.95
            }
        
        return {
            'prediction': prediction,
            'lower_bound': prediction * 0.95,  # Simplified
            'upper_bound': prediction * 1.05,  # Simplified
            'confidence': 0.90
        }
    
    def save_predictions(self, predictions: pd.DataFrame, symbol: str):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pred_path = os.path.join('data', 'predictions', f'{symbol}_{timestamp}.csv')
        
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        predictions.to_csv(pred_path)
        
        self.logger.info(f"Predictions saved to {pred_path}")

class Backtester:
    def __init__(self, model_path: str, config_path: str = 'config.yaml'):
        self.predictor = StockPredictor(model_path, config_path)
        self.logger = get_logger(__name__)
        
    def run(self, symbol: str, start_date: str, end_date: str,
           initial_capital: float = 10000) -> Dict[str, Any]:
        self.logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
        
        # Fetch historical data
        fetcher = StockDataFetcher()
        data = fetcher.fetch_single(symbol, start_date, end_date)
        
        if data.empty:
            raise ValueError("No data available for backtesting")
        
        # Initialize portfolio
        portfolio = {
            'cash': initial_capital,
            'shares': 0,
            'total_value': initial_capital,
            'trades': []
        }
        
        results = []
        
        # Simulate trading
        for i in range(100, len(data) - 1):  # Need history for predictions
            current_date = data.index[i]
            current_price = data.iloc[i]['Close']
            historical_data = data.iloc[:i+1]
            
            # Make prediction
            try:
                next_day_prediction = self.predictor.predict_next_day(symbol, historical_data)
            except Exception as e:
                self.logger.warning(f"Prediction failed for {current_date}: {str(e)}")
                continue
            
            # Simple trading strategy
            predicted_return = (next_day_prediction - current_price) / current_price
            
            # Buy signal
            if predicted_return > 0.01 and portfolio['shares'] == 0:  # 1% threshold
                shares_to_buy = int(portfolio['cash'] / current_price)
                if shares_to_buy > 0:
                    portfolio['shares'] = shares_to_buy
                    portfolio['cash'] -= shares_to_buy * current_price
                    portfolio['trades'].append({
                        'date': current_date,
                        'action': 'buy',
                        'price': current_price,
                        'shares': shares_to_buy
                    })
            
            # Sell signal
            elif predicted_return < -0.01 and portfolio['shares'] > 0:  # -1% threshold
                portfolio['cash'] += portfolio['shares'] * current_price
                portfolio['trades'].append({
                    'date': current_date,
                    'action': 'sell',
                    'price': current_price,
                    'shares': portfolio['shares']
                })
                portfolio['shares'] = 0
            
            # Calculate portfolio value
            portfolio['total_value'] = portfolio['cash'] + portfolio['shares'] * current_price
            
            results.append({
                'date': current_date,
                'portfolio_value': portfolio['total_value'],
                'price': current_price,
                'prediction': next_day_prediction,
                'shares': portfolio['shares'],
                'cash': portfolio['cash']
            })
        
        # Calculate final metrics
        final_value = portfolio['total_value']
        total_return = (final_value - initial_capital) / initial_capital
        
        results_df = pd.DataFrame(results)
        
        # Calculate Sharpe ratio (simplified)
        if len(results_df) > 1:
            daily_returns = results_df['portfolio_value'].pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(portfolio['trades']),
            'trades': portfolio['trades'],
            'results_df': results_df
        }

def main():
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Make stock price predictions')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--symbol', type=str, required=True,
                       help='Stock symbol')
    parser.add_argument('--days', type=int, default=5,
                       help='Number of days to predict')
    
    args = parser.parse_args()
    
    predictor = StockPredictor(args.model_path)
    
    # Make predictions
    predictions = predictor.predict_multiple_days(args.symbol, args.days)
    print("\nPredictions:")
    print(predictions)
    
    # Save predictions
    predictor.save_predictions(predictions, args.symbol)

if __name__ == "__main__":
    main()