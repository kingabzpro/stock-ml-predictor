#!/usr/bin/env python3
"""
Demo script to showcase the ML stock prediction system
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.fetch_data import StockDataFetcher
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.predict import StockPredictor

def main():
    print("=== Stock ML Prediction System Demo ===\n")
    
    # Configuration
    symbol = 'AAPL'
    algorithm = 'gradient_boosting'
    
    print(f"1. Fetching data for {symbol}...")
    fetcher = StockDataFetcher()
    data = fetcher.fetch_single(symbol, start='2020-01-01', end='2023-12-31')
    print(f"   ✓ Fetched {len(data)} days of data\n")
    
    print(f"2. Training {algorithm} model...")
    trainer = ModelTrainer()
    model = trainer.train(symbol, algorithm)
    print(f"   ✓ Model trained successfully\n")
    
    print("3. Evaluating model...")
    model_path = os.path.join('models', 'trained', f'{symbol}_{algorithm}_latest.pkl')
    evaluator = ModelEvaluator(model_path)
    
    # Get test data
    test_data = fetcher.fetch_single(symbol, start='2024-01-01', end='2024-05-20')
    
    if len(test_data) > 100:  # Only evaluate if we have enough data
        metrics = evaluator.generate_report(test_data)
        print("   ✓ Model evaluation metrics:")
        for metric, value in metrics.items():
            print(f"     - {metric.upper()}: {value:.4f}")
    else:
        print("   ⚠ Not enough test data for full evaluation")
        # Simple evaluation with available data
        from src.data.preprocess import DataPreprocessor
        preprocessor = DataPreprocessor()
        try:
            X_test, y_test = preprocessor.prepare_data(test_data)
            if len(X_test) > 0:
                metrics = evaluator.evaluate(X_test, y_test)
                print("   ✓ Model evaluation metrics (limited data):")
                for metric, value in metrics.items():
                    print(f"     - {metric.upper()}: {value:.4f}")
        except:
            print("   ⚠ Could not evaluate with limited data")
    print()
    
    print("4. Making predictions...")
    predictor = StockPredictor(model_path)
    
    try:
        # Single day prediction
        next_day_price = predictor.predict_next_day(symbol)
        print(f"   ✓ Next day predicted price: ${next_day_price:.2f}")
        
        # Multiple days prediction
        predictions = predictor.predict_multiple_days(symbol, days=5)
        print(f"   ✓ 5-day predictions:")
        print(predictions)
    except Exception as e:
        print(f"   ⚠ Prediction error: {str(e)}")
        print("   Note: This might happen if market data is not available")
    print()
    
    print("5. Starting API server...")
    print("   Run 'python src/api/app.py' to start the server")
    print("   API will be available at http://localhost:5000")
    print()
    
    print("Demo completed! Check the following directories:")
    print("   - data/raw/: Raw stock data")
    print("   - data/processed/: Processed features")
    print("   - models/trained/: Trained models")
    print("   - reports/: Evaluation reports")

if __name__ == "__main__":
    main()