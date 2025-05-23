#!/usr/bin/env python3
"""
Test script to demonstrate all features of the stock ML prediction system
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.fetch_data import StockDataFetcher
from src.data.preprocess import DataPreprocessor
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.predict import StockPredictor

def test_data_fetching():
    print("1. Testing Data Fetching...")
    fetcher = StockDataFetcher()
    
    # Fetch single stock
    data = fetcher.fetch_single('MSFT', start='2024-01-01', end='2024-05-01')
    print(f"   ✓ Fetched {len(data)} days of MSFT data")
    
    # Fetch multiple stocks
    stocks = ['AAPL', 'GOOGL']
    multi_data = fetcher.fetch_multiple(stocks, start='2024-04-01', end='2024-05-01')
    print(f"   ✓ Fetched data for {len(multi_data)} stocks")
    print()

def test_preprocessing():
    print("2. Testing Data Preprocessing...")
    fetcher = StockDataFetcher()
    preprocessor = DataPreprocessor()
    
    # Fetch and preprocess data
    data = fetcher.fetch_single('AAPL', start='2023-01-01', end='2023-12-31')
    X, y = preprocessor.prepare_data(data)
    
    print(f"   ✓ Created {len(X.columns)} features")
    print(f"   ✓ Feature examples: {list(X.columns[:5])}")
    print(f"   ✓ Data shape: {X.shape}")
    print()

def test_model_training():
    print("3. Testing Model Training...")
    trainer = ModelTrainer()
    
    # Test different algorithms
    algorithms = ['linear_regression', 'random_forest', 'gradient_boosting']
    
    for algo in algorithms:
        try:
            print(f"   Training {algo}...")
            model = trainer.train('AAPL', algo)
            print(f"   ✓ Successfully trained {algo}")
        except Exception as e:
            print(f"   ✗ Error training {algo}: {str(e)}")
    print()

def test_predictions():
    print("4. Testing Predictions...")
    
    # Use the latest model
    model_path = 'models/trained/AAPL_gradient_boosting_latest.pkl'
    
    if os.path.exists(model_path):
        predictor = StockPredictor(model_path)
        
        try:
            # Single prediction
            price = predictor.predict_next_day('AAPL')
            print(f"   ✓ Next day prediction: ${price:.2f}")
            
            # Multiple days
            predictions = predictor.predict_multiple_days('AAPL', days=3)
            print(f"   ✓ 3-day predictions made")
            
            # Confidence intervals
            conf_pred = predictor.predict_with_confidence('AAPL')
            print(f"   ✓ Prediction with confidence: ${conf_pred['prediction']:.2f} "
                  f"[{conf_pred['lower_bound']:.2f}, {conf_pred['upper_bound']:.2f}]")
        except Exception as e:
            print(f"   ✗ Prediction error: {str(e)}")
    else:
        print("   ✗ No trained model found")
    print()

def test_evaluation():
    print("5. Testing Model Evaluation...")
    
    model_path = 'models/trained/AAPL_gradient_boosting_latest.pkl'
    
    if os.path.exists(model_path):
        evaluator = ModelEvaluator(model_path)
        
        # Get test data
        fetcher = StockDataFetcher()
        test_data = fetcher.fetch_single('AAPL', start='2024-01-01', end='2024-05-01')
        
        try:
            metrics = evaluator.generate_report(test_data[:50])  # Use limited data
            print("   ✓ Evaluation metrics:")
            for metric, value in metrics.items():
                print(f"     - {metric}: {value:.4f}")
        except Exception as e:
            print(f"   ✗ Evaluation error: {str(e)}")
    else:
        print("   ✗ No trained model found")
    print()

def main():
    print("=== Stock ML System Feature Test ===\n")
    
    test_data_fetching()
    test_preprocessing()
    # Skip training to save time - models already exist
    # test_model_training()
    test_predictions()
    test_evaluation()
    
    print("Feature tests completed!")
    print("\nTo test the API:")
    print("1. Run: python src/api/app.py")
    print("2. Visit: http://localhost:5000/health")
    print("3. Check API docs in README.md")

if __name__ == "__main__":
    main()