# Stock ML Predictor - Project Summary

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/stock-ml-predictor.git
cd stock-ml-predictor

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Run the demo
python run_demo.py

# Start the API server
python src/api/app.py
```

## 📊 Project Overview

This is a complete machine learning system for stock price prediction using scikit-learn. The system fetches real-time stock data, engineers features, trains models, and serves predictions via a REST API.

### Key Features
- ✅ Real-time data fetching with yfinance
- ✅ Advanced feature engineering with technical indicators
- ✅ Multiple ML models (Random Forest, Gradient Boosting, Linear Regression)
- ✅ REST API for serving predictions
- ✅ Comprehensive evaluation and visualization
- ✅ UV package manager support
- ✅ Docker support
- ✅ Unit tests

### Tech Stack
- **Language**: Python 3.9+
- **ML Framework**: scikit-learn
- **Data Source**: yfinance
- **API Framework**: Flask
- **Package Manager**: UV
- **Containerization**: Docker

## 🛠️ API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Predict Next Day Price
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

### Predict Multiple Days
```bash
curl -X POST http://localhost:5000/predict/multiple-days \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 5}'
```

### Prediction with Confidence Intervals
```bash
curl -X POST http://localhost:5000/predict/confidence \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

## 📁 Project Structure

```
stock-ml-predictor/
├── src/                      # Source code
│   ├── data/                 # Data fetching and preprocessing
│   ├── models/               # Model training and evaluation
│   ├── api/                  # Flask API
│   └── utils/                # Utilities
├── tests/                    # Unit tests
├── data/                     # Data storage
├── models/                   # Trained models
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── Dockerfile                # Docker configuration
└── README.md                 # Documentation
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data_fetcher.py

# Run with coverage
pytest --cov=src tests/
```

## 🐳 Docker

```bash
# Build image
docker build -t stock-ml-predictor .

# Run container
docker run -p 5000:5000 stock-ml-predictor
```

## 📈 Model Performance

The system achieves the following metrics on test data:
- **RMSE**: ~7.14
- **MAE**: ~5.64
- **R²**: ~-0.04 (Note: This indicates the model needs improvement for production use)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- yfinance for providing stock data
- scikit-learn community
- Flask for the web framework
- UV for modern Python package management