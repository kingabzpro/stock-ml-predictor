import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger
from src.utils.config import Config
from src.data.preprocess import DataPreprocessor

class ModelEvaluator:
    def __init__(self, model_path: str, config_path: str = 'config.yaml'):
        self.config = Config(config_path)
        self.logger = get_logger(__name__, self.config.get_logging_config())
        self.model_path = model_path
        self.model = self._load_model()
        self.metadata = self._load_metadata()
        self.scaler = self._load_scaler()
        self.preprocessor = DataPreprocessor(config_path)
        self.preprocessor.scaler = self.scaler  # Set the loaded scaler
        
    def _load_model(self):
        return joblib.load(self.model_path)
    
    def _load_metadata(self):
        metadata_path = self.model_path.replace('.pkl', '_metadata.pkl')
        if os.path.exists(metadata_path):
            return joblib.load(metadata_path)
        return {}
    
    def _load_scaler(self):
        scaler_path = self.model_path.replace('.pkl', '_scaler.pkl')
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
        return None
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        self.logger.info("Evaluating model performance")
        
        # Scale test data
        if self.scaler:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_test_scaled = X_test
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def plot_predictions(self, y_true: pd.Series, y_pred: np.ndarray,
                        title: str = "Model Predictions vs Actual") -> go.Figure:
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=y_true.index,
            y=y_true.values,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=y_true.index,
            y=y_pred,
            mode='lines',
            name='Predictions',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_residuals(self, y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals Over Time', 'Residual Distribution',
                          'Q-Q Plot', 'Residuals vs Predicted')
        )
        
        # Residuals over time
        fig.add_trace(
            go.Scatter(x=y_true.index, y=residuals, mode='markers',
                      marker=dict(color='blue', size=5), name='Residuals'),
            row=1, col=1
        )
        
        # Residual distribution
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=50, name='Distribution'),
            row=1, col=2
        )
        
        # Q-Q plot
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                      mode='markers', name='Q-Q', marker=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                      mode='lines', name='Normal', line=dict(color='black', dash='dash')),
            row=2, col=1
        )
        
        # Residuals vs predicted
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers',
                      marker=dict(color='green', size=5), name='Residuals vs Predicted'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        return fig
    
    def plot_feature_importance(self) -> Optional[go.Figure]:
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            feature_names = self.preprocessor.get_feature_names()
            importances = self.model.feature_importances_
            
            # Sort by importance
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            fig = go.Figure([go.Bar(
                x=importances[indices],
                y=[feature_names[i] for i in indices],
                orientation='h'
            )])
            
            fig.update_layout(
                title='Top 20 Feature Importances',
                xaxis_title='Importance',
                yaxis_title='Feature'
            )
            
            return fig
        
        return None
    
    def generate_report(self, test_data: pd.DataFrame, output_path: str = 'reports/'):
        self.logger.info("Generating evaluation report")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Prepare test data
        X_test, y_test = self.preprocessor.prepare_data(test_data)
        
        # Evaluate
        metrics = self.evaluate(X_test, y_test)
        
        # Make predictions for plotting
        if self.scaler:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_test_scaled = X_test
        
        predictions = self.model.predict(X_test_scaled)
        
        # Create plots
        pred_fig = self.plot_predictions(y_test, predictions)
        residual_fig = self.plot_residuals(y_test, predictions)
        
        # Save plots
        timestamp = self.metadata.get('timestamp', 'unknown')
        symbol = self.metadata.get('symbol', 'unknown')
        algorithm = self.metadata.get('algorithm', 'unknown')
        
        pred_fig.write_html(os.path.join(output_path, f'{symbol}_{algorithm}_{timestamp}_predictions.html'))
        residual_fig.write_html(os.path.join(output_path, f'{symbol}_{algorithm}_{timestamp}_residuals.html'))
        
        # Feature importance for tree-based models
        importance_fig = self.plot_feature_importance()
        if importance_fig:
            importance_fig.write_html(os.path.join(output_path, f'{symbol}_{algorithm}_{timestamp}_importance.html'))
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(output_path, f'{symbol}_{algorithm}_{timestamp}_metrics.csv'), index=False)
        
        self.logger.info(f"Report saved to {output_path}")
        
        return metrics
    
    def compare_models(self, model_paths: list) -> pd.DataFrame:
        results = []
        
        for model_path in model_paths:
            evaluator = ModelEvaluator(model_path)
            metadata = evaluator.metadata
            
            result = {
                'model_path': model_path,
                'symbol': metadata.get('symbol', 'unknown'),
                'algorithm': metadata.get('algorithm', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown')
            }
            
            # Add metrics
            metrics = metadata.get('metrics', {})
            result.update(metrics)
            
            results.append(result)
        
        comparison_df = pd.DataFrame(results)
        return comparison_df.sort_values('rmse')

def main():
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--symbol', type=str, required=True,
                       help='Stock symbol for test data')
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                       help='Start date for test data')
    parser.add_argument('--end_date', type=str, default='2024-01-01',
                       help='End date for test data')
    
    args = parser.parse_args()
    
    # Load test data
    from src.data.fetch_data import StockDataFetcher
    fetcher = StockDataFetcher()
    test_data = fetcher.fetch_single(args.symbol, args.start_date, args.end_date)
    
    # Evaluate
    evaluator = ModelEvaluator(args.model_path)
    evaluator.generate_report(test_data)

if __name__ == "__main__":
    main()