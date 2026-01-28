# Mitsui Commodity Price Prediction

Time-series forecasting project for commodity price prediction using machine learning, developed for the [Mitsui Bussan Commodities Kaggle Competition](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge).

## Competition Overview

Predicting commodity prices across multiple asset classes with complex temporal dependencies and market regime changes. There are are 424 price return targets to predict which entails commodities, equities and their differentials. Each target has a holding period from 1 to 4 days.

## My Approach

### Models Explored
Multi-layer LSTM models with drop-out.

### Feature Engineering
1000+ features were generated for 106 commodity / equity underlyings, including: 
- Lagged price features
- Rolling statistics (mean, std, min, max)
- Technical indicators (RSI, MACD)
- Volatility measures

### Feature Selection
- Top 300 out of the 1000 features were selected at the first stage using Mutual Information
- Random Forest regressor implemented using the 300 features
- Top 100 out of the 300 were selected (based in feature importance) as ultimate features for LSTM

### Data Preprocessing
- Missing values were mainly filled forward 
- Outliers were untouched
- Standard scaling applied
- Train-test split for time-series as per the competition

## Technical Skills Demonstrated

- Time-series forecasting with commodity and equity price data
- Feature engineering for financial markets
- LSTM implementation
- Handling multi-asset temporal data
- Data preprocessing for sequential data

## Challenges & Learnings

This project presented interesting technical challenges:

**Submission Technical Issue:**
Encountered persistent Kaggle submission errors during the competition that prevented final scoring. This highlighted the importance of:
- Early submission testing with validation data
- Understanding competition API requirements thoroughly
- Building robust error handling into submission pipelines
- Unfortunately never really knew what caused the "Kaggle error" during submission even though i enquired a lot. My "Save & Commit" were giving successful results. I guess the culprit is the CPU parallelization which i added in the prediction function, to safely stay within the 5-minutes prediction timlines.

**Key Takeaways:**
This was my first ever Kaggle competition, with valuable learning experience:
- LSTMs can give really powerful results , successfully modelling the temporal dependencies when there are several high quality external factor time series available as features. I observed Sharpe ratios of over 1.5 on several validation sets.
- Make sure to train for enough number of epochs to allow these competent LSTM models to learn. My experience shows at least 20+ epochs were needed in this case.
- I would test the submission pipeline / inference API early to identify problems

Despite the submission issue, this project strengthened my understanding of time-series modeling, feature engineering for financial data, and the complexities of real-world forecasting challenges.

## Dataset

Mitsui Bussan Commodities competition data featuring:
- Multiple commodity and equity price time-series spanning circa 8 years.

**Competition Link:** https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge

## Project Structure

