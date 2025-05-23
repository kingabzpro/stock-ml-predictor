import os
import glob
import json
from flask import Flask, request, jsonify
from typing import Dict, Any
import pandas as pd

from src.utils.logger import get_logger
from src.models.predict import StockPredictor
from src.models.train import ModelTrainer
from src.data.fetch_data import StockDataFetcher

def create_endpoints(app: Flask, config: Any):
    logger = get_logger(__name__, config.get_logging_config())
    
    # Initialize model predictor (load latest model)
    model_path = None
    trained_models_path = 'models/trained/'
    
    # Find latest model
    if os.path.exists(trained_models_path):
        model_files = glob.glob(os.path.join(trained_models_path, '*_latest.pkl'))
        if model_files:
            model_path = model_files[0]
            logger.info(f"Loaded model: {model_path}")
    
    predictor = StockPredictor(model_path) if model_path else None
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            symbol = data.get('symbol')
            features = data.get('features')
            
            if not symbol:
                return jsonify({'error': 'Symbol is required'}), 400
            
            if not predictor:
                return jsonify({'error': 'No model loaded'}), 500
            
            # If features provided, use them
            if features:
                prediction = predictor.predict_batch({symbol: features})[symbol]
            else:
                # Otherwise, fetch latest data and predict
                prediction = predictor.predict_next_day(symbol)
            
            return jsonify({
                'symbol': symbol,
                'prediction': prediction,
                'currency': 'USD'
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/predict/batch', methods=['POST'])
    def predict_batch():
        try:
            data = request.get_json()
            
            if not data or 'predictions' not in data:
                return jsonify({'error': 'Invalid request format'}), 400
            
            if not predictor:
                return jsonify({'error': 'No model loaded'}), 500
            
            # Extract features for each symbol
            features_dict = {}
            for item in data['predictions']:
                symbol = item.get('symbol')
                features = item.get('features')
                
                if symbol and features:
                    features_dict[symbol] = features
            
            # Make batch predictions
            predictions = predictor.predict_batch(features_dict)
            
            # Format response
            results = []
            for symbol, prediction in predictions.items():
                results.append({
                    'symbol': symbol,
                    'prediction': prediction,
                    'currency': 'USD'
                })
            
            return jsonify({'predictions': results})
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/predict/multiple-days', methods=['POST'])
    def predict_multiple_days():
        try:
            data = request.get_json()
            
            symbol = data.get('symbol')
            days = data.get('days', 5)
            
            if not symbol:
                return jsonify({'error': 'Symbol is required'}), 400
            
            if not predictor:
                return jsonify({'error': 'No model loaded'}), 500
            
            # Make predictions
            predictions_df = predictor.predict_multiple_days(symbol, days)
            
            # Convert to JSON-friendly format
            predictions = []
            for date, row in predictions_df.iterrows():
                predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_price': row['Predicted_Price']
                })
            
            return jsonify({
                'symbol': symbol,
                'predictions': predictions,
                'currency': 'USD'
            })
            
        except Exception as e:
            logger.error(f"Multiple days prediction error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/predict/confidence', methods=['POST'])
    def predict_with_confidence():
        try:
            data = request.get_json()
            
            symbol = data.get('symbol')
            
            if not symbol:
                return jsonify({'error': 'Symbol is required'}), 400
            
            if not predictor:
                return jsonify({'error': 'No model loaded'}), 500
            
            # Make prediction with confidence
            result = predictor.predict_with_confidence(symbol)
            
            return jsonify({
                'symbol': symbol,
                'prediction': result['prediction'],
                'confidence_interval': {
                    'lower': result['lower_bound'],
                    'upper': result['upper_bound'],
                    'confidence': result['confidence']
                },
                'currency': 'USD'
            })
            
        except Exception as e:
            logger.error(f"Confidence prediction error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/model/info', methods=['GET'])
    def model_info():
        try:
            if not predictor:
                return jsonify({'error': 'No model loaded'}), 500
            
            metadata = predictor.metadata
            
            return jsonify({
                'model_path': predictor.model_path,
                'symbol': metadata.get('symbol', 'unknown'),
                'algorithm': metadata.get('algorithm', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown'),
                'metrics': metadata.get('metrics', {}),
                'config': metadata.get('config', {})
            })
            
        except Exception as e:
            logger.error(f"Model info error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/model/list', methods=['GET'])
    def list_models():
        try:
            models = []
            
            if os.path.exists(trained_models_path):
                model_files = glob.glob(os.path.join(trained_models_path, '*.pkl'))
                
                for model_file in model_files:
                    if '_scaler' not in model_file and '_metadata' not in model_file:
                        # Load metadata
                        metadata_file = model_file.replace('.pkl', '_metadata.pkl')
                        if os.path.exists(metadata_file):
                            import joblib
                            metadata = joblib.load(metadata_file)
                            
                            models.append({
                                'filename': os.path.basename(model_file),
                                'path': model_file,
                                'symbol': metadata.get('symbol', 'unknown'),
                                'algorithm': metadata.get('algorithm', 'unknown'),
                                'timestamp': metadata.get('timestamp', 'unknown'),
                                'metrics': metadata.get('metrics', {})
                            })
            
            return jsonify({'models': models})
            
        except Exception as e:
            logger.error(f"List models error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/model/load', methods=['POST'])
    def load_model():
        try:
            global predictor
            
            data = request.get_json()
            model_path = data.get('model_path')
            
            if not model_path:
                return jsonify({'error': 'Model path is required'}), 400
            
            if not os.path.exists(model_path):
                return jsonify({'error': 'Model file not found'}), 404
            
            # Load new model
            predictor = StockPredictor(model_path)
            
            return jsonify({
                'message': 'Model loaded successfully',
                'model_path': model_path
            })
            
        except Exception as e:
            logger.error(f"Load model error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/train', methods=['POST'])
    def train_model():
        try:
            data = request.get_json()
            
            symbol = data.get('symbol')
            algorithm = data.get('algorithm', 'xgboost')
            
            if not symbol:
                return jsonify({'error': 'Symbol is required'}), 400
            
            # Initialize trainer
            trainer = ModelTrainer(config.config_path)
            
            # Train model
            logger.info(f"Starting training for {symbol} with {algorithm}")
            model = trainer.train(symbol, algorithm)
            
            # Reload predictor with new model
            global predictor
            latest_model_path = os.path.join(trained_models_path, f'{symbol}_{algorithm}_latest.pkl')
            if os.path.exists(latest_model_path):
                predictor = StockPredictor(latest_model_path)
            
            return jsonify({
                'message': 'Model trained successfully',
                'symbol': symbol,
                'algorithm': algorithm
            })
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/data/fetch', methods=['POST'])
    def fetch_data():
        try:
            data = request.get_json()
            
            symbol = data.get('symbol')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            
            if not symbol:
                return jsonify({'error': 'Symbol is required'}), 400
            
            # Fetch data
            fetcher = StockDataFetcher(config.config_path)
            stock_data = fetcher.fetch_single(symbol, start_date, end_date)
            
            if stock_data.empty:
                return jsonify({'error': 'No data found'}), 404
            
            # Convert to JSON-friendly format
            data_json = stock_data.reset_index().to_dict('records')
            
            return jsonify({
                'symbol': symbol,
                'data': data_json,
                'count': len(data_json)
            })
            
        except Exception as e:
            logger.error(f"Data fetch error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/data/info', methods=['GET'])
    def data_info():
        try:
            symbol = request.args.get('symbol')
            
            if not symbol:
                return jsonify({'error': 'Symbol is required'}), 400
            
            # Get stock info
            fetcher = StockDataFetcher(config.config_path)
            info = fetcher.get_info(symbol)
            
            return jsonify(info)
            
        except Exception as e:
            logger.error(f"Data info error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    logger.info("API endpoints created successfully")