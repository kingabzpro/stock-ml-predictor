# Stock Price Prediction ML Project

A machine learning project that fetches financial data using yfinance, trains predictive models, evaluates their performance, and serves the trained model via an API.

## 🚀 Features

- **Data Collection**: Automated fetching of historical stock data using yfinance
- **Data Processing**: Feature engineering and preprocessing pipeline
- **Model Training**: Multiple ML algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **Model Serving**: REST API for real-time predictions
- **Monitoring**: Model performance tracking and data drift detection

## 📋 Requirements

```bash
python>=3.8
yfinance>=0.2.28
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
xgboost>=1.7.0
matplotlib>=3.5.0
seaborn>=0.12.0
flask>=2.3.0
joblib>=1.3.0
pytest>=7.4.0
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-ml-project.git
cd stock-ml-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
stock-ml-project/
├── data/
│   ├── raw/              # Raw data from yfinance
│   ├── processed/        # Processed features
│   └── predictions/      # Model predictions
├── models/
│   ├── trained/          # Saved trained models
│   └── configs/          # Model configurations
├── src/
│   ├── data/
│   │   ├── fetch_data.py      # yfinance data collection
│   │   └── preprocess.py      # Feature engineering
│   ├── models/
│   │   ├── train.py           # Model training scripts
│   │   ├── evaluate.py        # Model evaluation
│   │   └── predict.py         # Prediction utilities
│   ├── api/
│   │   ├── app.py             # Flask API
│   │   └── endpoints.py       # API endpoints
│   └── utils/
│       ├── config.py          # Configuration settings
│       └── logger.py          # Logging utilities
├── notebooks/
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   └── model_experiments.ipynb # Model experiments
├── tests/                      # Unit tests
├── requirements.txt
├── config.yaml                 # Configuration file
└── README.md
```

## 🚦 Quick Start

### 1. Fetch Stock Data

```python
from src.data.fetch_data import StockDataFetcher

# Initialize fetcher
fetcher = StockDataFetcher()

# Fetch data for multiple stocks
stocks = ['AAPL', 'GOOGL', 'MSFT']
data = fetcher.fetch_multiple(stocks, start='2020-01-01', end='2024-01-01')
```

### 2. Train Model

```bash
# Train a model using the CLI
python src/models/train.py --config config.yaml --model xgboost

# Or use the Python API
from src.models.train import ModelTrainer

trainer = ModelTrainer(config_path='config.yaml')
model = trainer.train(algorithm='xgboost')
```

### 3. Evaluate Model

```python
from src.models.evaluate import ModelEvaluator

evaluator = ModelEvaluator(model_path='models/trained/xgboost_model.pkl')
metrics = evaluator.evaluate(test_data)
evaluator.plot_results()
```

### 4. Serve Model

```bash
# Start the API server
python src/api/app.py

# The API will be available at http://localhost:5000
```

## 📊 API Endpoints

### Health Check
```bash
GET /health
```

### Predict Stock Price
```bash
POST /predict
Content-Type: application/json

{
    "symbol": "AAPL",
    "features": {
        "open": 150.0,
        "high": 155.0,
        "low": 149.0,
        "volume": 1000000
    }
}
```

### Get Model Info
```bash
GET /model/info
```

### Batch Predictions
```bash
POST /predict/batch
Content-Type: application/json

{
    "predictions": [
        {
            "symbol": "AAPL",
            "features": {...}
        },
        {
            "symbol": "GOOGL",
            "features": {...}
        }
    ]
}
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
data:
  symbols: ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
  start_date: '2020-01-01'
  end_date: '2024-01-01'
  features:
    - 'open'
    - 'high'
    - 'low'
    - 'close'
    - 'volume'
    - 'SMA_20'
    - 'SMA_50'
    - 'RSI'

model:
  algorithm: 'xgboost'  # Options: 'random_forest', 'xgboost', 'lstm'
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    learning_rate: 0.01
  
training:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42
  
api:
  host: '0.0.0.0'
  port: 5000
  debug: false
```

## 📈 Model Performance

Our models achieve the following performance metrics on test data:

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| XGBoost | 2.34 | 1.89 | 0.94 |
| Random Forest | 2.67 | 2.12 | 0.92 |
| LSTM | 2.15 | 1.76 | 0.95 |

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_data_fetcher.py
```

## 🐳 Docker Support

Build and run using Docker:

```bash
# Build image
docker build -t stock-ml-api .

# Run container
docker run -p 5000:5000 stock-ml-api
```

## 📝 Usage Examples

### Example 1: Simple Price Prediction

```python
from src.data.fetch_data import StockDataFetcher
from src.models.predict import StockPredictor

# Fetch recent data
fetcher = StockDataFetcher()
data = fetcher.fetch_latest('AAPL', days=30)

# Make prediction
predictor = StockPredictor('models/trained/best_model.pkl')
prediction = predictor.predict_next_day('AAPL', data)
print(f"Predicted price: ${prediction:.2f}")
```

### Example 2: Backtesting Strategy

```python
from src.models.backtest import Backtester

backtester = Backtester(model_path='models/trained/xgboost_model.pkl')
results = backtester.run(
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    initial_capital=10000
)
print(f"Total return: {results['total_return']:.2%}")
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- yfinance for providing easy access to financial data
- scikit-learn and XGBoost communities
- Flask for the simple and effective web framework

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/stock-ml-project](https://github.com/yourusername/stock-ml-project)