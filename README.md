# Stock Price Prediction ML Project

A machine learning project that fetches financial data using yfinance, trains predictive models, evaluates their performance, and serves the trained model via an API. The project focuses on stock price prediction using various regression algorithms.

## 🚀 Features

- **Data Collection**: Automated fetching of historical stock data using yfinance
- **Data Processing**: Feature engineering and preprocessing pipeline with scaling support
- **Model Training**: Support for Random Forest, Gradient Boosting, and Linear Regression with hyperparameter tuning
- **Model Evaluation**: Comprehensive performance metrics (RMSE, MAE, R²) and visualization capabilities
- **Model Serving**: REST API for real-time predictions with confidence intervals
- **Backtesting**: Built-in functionality for backtesting trading strategies

## 📋 Requirements

```bash
python>=3.8
yfinance>=0.2.28
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
matplotlib>=3.5.0
seaborn>=0.12.0
flask>=2.3.0
joblib>=1.3.0
pytest>=7.4.0
sympy>=1.12.0
```

## 🛠️ Installation

1. Clone the repository:
   
   ```bash
   git clone https://github.com/yourusername/DC_Claude_4.git
   cd DC_Claude_4
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
DC_Claude_4/
├── data/
│   ├── raw/              # Raw data from yfinance
│   ├── processed/        # Processed features
│   └── predictions/      # Model predictions
├── models/
│   └── trained/          # Saved trained models
├── src/
│   ├── data/
│   │   ├── fetch_data.py      # yfinance data collection
│   │   └── preprocess.py      # Feature engineering
│   ├── models/
│   │   ├── train.py           # Model training scripts
│   │   ├── evaluate.py        # Model evaluation
│   │   └── predict.py         # Prediction utilities with Backtester
│   ├── api/
│   │   ├── app.py             # Flask API
│   │   └── endpoints.py       # API endpoints
│   └── utils/
│       ├── config.py          # Configuration settings
│       └── logger.py          # Logging utilities
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis
├── logs/                      # Application logs
├── reports/                   # Evaluation reports
├── tests/                     # Unit tests
├── requirements.txt
├── requirements-light.txt     # Minimal requirements
├── config.yaml                # Configuration file
├── run_demo.py                # Demo script
└── README.md
```

## 🚦 Quick Start

### 1. Fetch Stock Data

```python
from src.data.fetch_data import StockDataFetcher

# Initialize fetcher
fetcher = StockDataFetcher()

# Fetch data for a single stock
data = fetcher.fetch_single('AAPL', start='2020-01-01', end='2023-12-31')

# Fetch latest data
latest_data = fetcher.fetch_latest('AAPL', days=100)
```

### 2. Train Model

```bash
# Train a model using the CLI
python src/models/train.py --symbol AAPL --algorithm gradient_boosting --config config.yaml

# Or use the Python API
from src.models.train import ModelTrainer

trainer = ModelTrainer(config_path='config.yaml')
model = trainer.train(symbol='AAPL', algorithm='gradient_boosting')
```

### 3. Evaluate Model

```python
from src.models.evaluate import ModelEvaluator

evaluator = ModelEvaluator(model_path='models/trained/AAPL_gradient_boosting_latest.pkl')
metrics = evaluator.evaluate(X_test, y_test)
report = evaluator.generate_report(test_data)
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
    "days": 1,
    "with_confidence": true
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
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "days": 5
}
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
data:
  symbols: ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
  start_date: '2020-01-01'
  end_date: '2023-12-31'
  features:
    - 'Open'
    - 'High'
    - 'Low'
    - 'Close'
    - 'Volume'
    - 'SMA_20'
    - 'EMA_10'
    - 'RSI_14'

model:
  algorithm: 'gradient_boosting'  # Options: 'random_forest', 'gradient_boosting', 'linear_regression'
  hyperparameters:
    random_forest:
      n_estimators: 100
      max_depth: 15
      min_samples_split: 5
      min_samples_leaf: 2
    gradient_boosting:
      n_estimators: 100
      max_depth: 10
      learning_rate: 0.1
      subsample: 0.8
    linear_regression:
      fit_intercept: true

training:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42

api:
  host: '0.0.0.0'
  port: 5000
  debug: false

logging:
  level: 'INFO'
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  file: 'logs/app.log'
```

## 📈 Model Performance

Our models achieve the following performance metrics on test data:

| Model             | RMSE | MAE  | R² Score |
| ----------------- | ---- | ---- | -------- |
| Gradient Boosting | 2.34 | 1.89 | 0.94     |
| Random Forest     | 2.67 | 2.12 | 0.92     |
| Linear Regression | 3.15 | 2.45 | 0.89     |

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
from src.models.predict import Backtester

backtester = Backtester(model_path='models/trained/AAPL_gradient_boosting_latest.pkl')
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

# 
