# Regression
Regression in machine learning is a supervised learning technique used to predict continuous, numerical output values (e.g., prices, temperature, sales) based on the relationship between input features (independent variables) and a target variable (dependent variable).

Understood. We’ll **stop adding features** and instead **close the project properly**.

Below is a **complete, clean technical report** that explains:

* what you built
* why the first model failed
* how you improved it
* how the evaluation metrics changed
* what the final results mean

You can use this:

* directly in your notebook as Markdown
* or adapt it for a README / report submission

---

# 📊 Semiconductor Stock Price Forecasting Using Linear Regression

**Case Study: NVIDIA (NVDA)**

---

## 1. Project Overview

This project explores the use of **Linear Regression** to forecast short-term semiconductor stock prices, using NVIDIA (NVDA) as a representative technology company. The objective is not to build a trading system, but to **understand how historical price information can be modeled using supervised learning**, and how proper feature design and evaluation significantly affect model performance.

---

## 2. Data Collection

Stock price data was sourced using the `yfinance` Python library, which retrieves historical market data from Yahoo Finance.

**Data included:**

* Date
* Open, High, Low, Close prices
* Trading Volume

For this project:

* Only **recent historical data** was used to avoid outdated market behavior.
* The **Close price** was selected as the target variable, representing the market’s final consensus price for each trading day.

---

## 3. Initial Model Attempt and Limitations

### Initial Approach

The first model attempted to predict stock prices using a naive linear relationship over time.

### Results (Initial Model)

* **MAE ≈ 11.75**
* **RMSE ≈ 14.00**
* **R² ≈ -0.22**

### Interpretation

* Errors were large, meaning predictions were far from actual prices.
* A **negative R² score** indicated that the model performed worse than simply predicting the average price.
* This showed that **time alone is not a sufficient predictor** for stock prices.

### Key Learning

> Financial time-series data does not follow a simple linear trend over long periods.

---

## 4. Model Improvement Strategy

To address the shortcomings, the modeling approach was refined by aligning it with the **temporal nature of financial data**.

### Key Improvements Made

1. **Shortened the Time Window**

   * Only the most recent data points were used.
   * This reduced noise from older market regimes.

2. **Introduced Lag-Based Feature Engineering**

   * A new feature, `prev_price`, was created:

     * Yesterday’s closing price used to predict today’s price.
   * This captures the strong **autocorrelation** present in stock prices.

3. **Time-Aware Train/Test Split**

   * Data was split without shuffling.
   * Prevented future information from leaking into training.

---

## 5. Final Model Performance

### Evaluation Metrics (Improved Model)

* **MAE ≈ 0.81**
* **RMSE ≈ 1.04**
* **R² ≈ 0.85**

### Metric Interpretation

* **MAE**: On average, predictions were within less than one price unit of actual values.
* **RMSE**: Large errors were rare, indicating stable predictions.
* **R² Score**: The model explained approximately **85% of the variance** in recent price movements.

### Improvement Summary

| Metric | Initial Model | Improved Model |
| ------ | ------------- | -------------- |
| MAE    | ~11.75        | ~0.81          |
| RMSE   | ~14.00        | ~1.04          |
| R²     | -0.22         | 0.85           |

This dramatic improvement highlights the importance of **feature selection over algorithm complexity**.

---

## 6. Visualization Analysis

A comparison plot of **actual vs predicted prices** showed that:

* The predicted line closely follows the actual price trend.
* The model accurately captures upward and downward movements.
* Slight lag is observed during sudden price changes.

### Explanation of Lag

The model uses **yesterday’s price** to predict today’s price.
As a result:

* Sudden market jumps or drops are reflected one step later.
* This behavior is expected and is a known limitation of linear, lag-based models.

This lag is **not a bug**, but a structural property of the model.

---

## 7. Model Strengths and Limitations

### Strengths

* Simple and interpretable
* Strong performance on short-term trends
* Demonstrates realistic financial modeling assumptions

### Limitations

* Cannot predict sudden news-driven events
* Not suitable for long-term forecasting
* Not intended for live trading decisions

---

## 8. Conclusion

This project demonstrates that even simple models like Linear Regression can perform well on financial data **when designed correctly**. By respecting time dependencies and applying appropriate feature engineering, the model achieved strong predictive performance.

The primary learning outcome is that **understanding the data structure and evaluation process is more important than using complex algorithms**.

---
