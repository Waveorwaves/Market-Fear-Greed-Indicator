# Market Fear & Greed Indicator ML Pipeline

Machine learning pipeline for predicting market sentiment and VIX volatility using financial data.

## 🎯 Project Overview

Predicts market sentiment (Fear/Greed/Stable) and VIX values using three feature engineering approaches:
- **V1**: Basic returns (20 features)
- **V2**: Returns + moving averages (30 features) 
- **V3**: Full feature set with ratios (50+ features)

## 🏆 Best Models by Version

### Classification (Market Sentiment)
- **V1**: **Random Forest** - 64.55% accuracy
- **V2**: **FLAML (LightGBM)** - 86.36% accuracy ⭐
- **V3**: **Random Forest** - 91.36% accuracy 🥇 **BEST**

### Regression (VIX Prediction)  
- **V1**: **Random Forest** - R² = 0.21
- **V2**: **Random Forest** - R² = 0.29
- **V3**: **Random Forest** - R² = 0.39 🥇 **BEST**

*Note: Original notebook approach used **Histogram Gradient Boosting (HGB)** which also performed excellently*

## 🔧 Key Files

- `Market_Fear_Greed_Indicator_Model.ipynb` - Original modeling (HGB focus)
- `automl_integration.py` - AutoML pipeline
- `feature_store_setup.py` - Feature engineering
- `../Integrated_Fear_Greed_AutoML_Pipeline.ipynb` - Complete pipeline

## 🚀 Quick Start

```python
# Run integrated pipeline
jupyter notebook ../Integrated_Fear_Greed_AutoML_Pipeline.ipynb

# Or run original approach
jupyter notebook Market_Fear_Greed_Indicator_Model.ipynb
```

## 📈 Key Insights

- **Moving averages** provide major performance boost (V1→V2: +22% accuracy)
- **Asset ratios** further improve results (V2→V3: +5% accuracy)  
- **Random Forest** dominates AutoML experiments (4/6 wins)
- **FLAML (LightGBM)** excellent for automated hyperparameter tuning
- **V3 features** optimal for both classification and regression

## 📁 Outputs

- **Models**: `../models/best_*.joblib` - Production-ready trained models
- **Results**: `../results/` - Experiment logs and analysis
- **Data**: `../data/` - Processed feature sets