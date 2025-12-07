# GOOGL Stock Market Prediction

Predicting next-day GOOGL price movements using machine learning with technical indicators and market sentiment data.

## Project Overview

**Theme:** Stock Market / Finance  
**Goal:** Binary classification - predict whether GOOGL stock price will go UP (1) or DOWN (0) the next trading day

This end-to-end data science project covers:
- **Part 1 - Extract:** Dataset creation from multiple APIs with feature engineering
- **Part 2 - Learn:** Model training and evaluation (Decision Tree, Random Forest, XGBoost)
- **Part 3 - Predict:** Prediction demo using the best model

## Data Sources

| Source | Data | API |
|--------|------|-----|
| Yahoo Finance | GOOGL stock price, volume (2 years) | `yfinance` |
| Federal Reserve (FRED) | VIX Volatility Index | `fredapi` |

**Dataset:** 497 samples (Nov 2023 - Nov 2025)

## Feature Engineering

| Feature | Type | Description |
|---------|------|-------------|
| Returns | Price Momentum | Daily returns percentage |
| Volume_Change | Market Activity | Volume change rate |
| High_Low_Ratio | Volatility | Intraday price range |
| VIX | Market Fear | CBOE Volatility Index |
| VIX_Change_5d | Sentiment Trend | 5-day VIX change |
| Short_Trend_Signal | Technical | Price above 5-day MA (1/0) |

## Model Results

| Model | Accuracy | F1-Score | Precision |
|-------|----------|----------|-----------|
| Decision Tree | 0.53 | 0.58 | 0.63 |
| Random Forest | 0.53 | 0.62 | 0.60 |
| **XGBoost** | **0.59** | **0.68** | **0.64** |

**Best Model:** XGBoost with hyperparameter tuning
- Best parameters: `learning_rate=0.01`, `max_depth=3`, `n_estimators=150`
- Sample prediction accuracy: 60% (6/10 correct on held-out data)

## Project Structure

```
├── Data/
│   ├── GOOGL_2y_Dataset.csv      # Main dataset
│   └── sample_data.csv           # Sample for prediction demo
├── Notebooks/
│   └── GOOGL_Stock_Prediction.ipynb  # Main notebook (EDA + Modeling + Prediction)
├── Model/
│   └── xgboost_best_model.pkl    # Saved best model
├── Slides/
│   └── presentation.pdf          # Project presentation
├── summary/
│   └── one_page_summary.pdf      # One-page observation summary
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/18201550122/GOOGL-Stock-Prediction.git
cd GOOGL-Stock-Prediction

# Install dependencies
pip install yfinance fredapi pandas numpy xgboost scikit-learn matplotlib
```

## Usage

### Run the Notebook
```bash
jupyter notebook Notebooks/GOOGL_Stock_Prediction.ipynb
```

### Quick Prediction Demo
```python
import pandas as pd
import pickle

# Load model
with open('Model/xgboost_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load sample data
data = pd.read_csv('Data/sample_data.csv', index_col='Date')

# Make predictions
predictions = model.predict(data)
print(predictions)  # 1 = UP, 0 = DOWN
```

## Key Findings

1. **XGBoost outperforms** Decision Tree and Random Forest in all metrics
2. **Low learning rate (0.01)** works best for noisy stock data
3. **Shallow trees (depth=3)** prevent overfitting
4. Model achieves **59% accuracy** (9% above random baseline)

## Limitations

- Limited to technical indicators and VIX; no fundamental or news data
- Single stock (GOOGL) - generalization not verified
- 497 samples may be insufficient for complex patterns
- Stock market inherently difficult to predict (Efficient Market Hypothesis)

## Future Improvements

- Add news sentiment analysis (Twitter/News API)
- Include more technical indicators (RSI, MACD, Bollinger Bands)
- Test LSTM or other time-series deep learning models
- Validate on multiple stocks
- Build Streamlit interactive dashboard

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance
- fredapi
- scikit-learn
- xgboost
- matplotlib

## License

MIT License

## Author

Yi Zhang - Northeastern University  
Songnan Zhao - Northeastern University  
Huizhen Zheng - Northeastern University  
