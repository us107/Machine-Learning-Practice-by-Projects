# ⚡ Elastic Net Regression — Complete Study Guide

> Combines L1 (Lasso) + L2 (Ridge) Regularization  
> Everything you need to understand, implement, and evaluate Elastic Net.

---

## 📚 Table of Contents

1. [What is Regularization?](#1-what-is-regularization)
2. [The Elastic Net Formula](#2-the-elastic-net-formula)
3. [L1 vs L2 vs Elastic Net](#3-l1-vs-l2-vs-elastic-net)
4. [Step-by-Step Code](#4-step-by-step-code)
5. [Results Interpretation](#5-results-interpretation)
6. [When to Use / Not Use](#6-when-to-use--not-use)
7. [Key Rules & Mental Model](#7-key-rules--mental-model)

---

## 1. What is Regularization?

**Problem:** Standard linear regression minimizes only RSS (Residual Sum of Squares). With many features or noisy data the model **overfits** — it memorizes noise instead of learning the true pattern.

**Solution:** Add a **penalty** on large coefficients to the loss function. This discourages the model from becoming too complex.

> Think of it as a budget — you can use large weights, but they cost you in the loss function.

---

## 2. The Elastic Net Formula

```
Loss = RSS  +  α·ρ·Σ|β|  +  α·(1-ρ)/2·Σβ²
               ↑ L1 penalty    ↑ L2 penalty
```

| Symbol | Name | Meaning |
|---|---|---|
| **α (alpha)** | Regularization strength | Bigger = more penalty = simpler model |
| **ρ (l1_ratio)** | L1 vs L2 mix | 0 = pure Ridge, 1 = pure Lasso, 0.5 = balanced |
| **β (beta)** | Coefficients | The model weights being penalized |

```python
def elastic_net_loss(y_true, y_pred, beta, alpha=1.0, l1_ratio=0.5):
    rss    = np.sum((y_true - y_pred) ** 2)                # fit quality
    l1_pen = alpha * l1_ratio * np.sum(np.abs(beta))        # sparsity
    l2_pen = alpha * (1 - l1_ratio) / 2 * np.sum(beta**2)  # shrinkage
    return rss + l1_pen + l2_pen
```

---

## 3. L1 vs L2 vs Elastic Net

| | Ridge (L2) | Lasso (L1) | Elastic Net |
|---|---|---|---|
| Penalty | α·Σβ² | α·Σ\|β\| | α·(ρ\|β\| + (1-ρ)β²) |
| Zeros in coefficients | ❌ Never | ✅ Yes | ✅ Yes |
| Correlated features | ✅ Handles well | ❌ Picks one randomly | ✅ Groups them |
| Feature selection | ❌ No | ✅ Yes | ✅ Yes |
| l1_ratio value | 0.0 | 1.0 | 0 to 1 |
| Best for | All features matter | Few strong features | Unknown / mixed |

---

## 4. Step-by-Step Code

### Step 1 — Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

---

### Step 2 — Generate Data

```python
X, y, true_coef = make_regression(
    n_samples=200,      # 200 rows
    n_features=50,      # 50 features total
    n_informative=10,   # only 10 actually matter
    noise=20,
    coef=True,
    random_state=42
)

print("X shape:", X.shape)   # (200, 50)
print("y shape:", y.shape)   # (200,)
```

> 50 features but only 10 are truly useful. Elastic Net should zero out the other 40.

---

### Step 3 — Split & Scale

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit on train only!
X_test  = scaler.transform(X_test)       # use train's stats on test
```

> ⚠️ Always scale before Elastic Net. The penalty unfairly punishes features with large numeric scales if you don't.

---

### Step 4 — Fit with Sklearn

```python
model = ElasticNet(
    alpha=1.0,       # regularization strength
    l1_ratio=0.5,    # 50% L1, 50% L2
    max_iter=5000
)
model.fit(X_train, y_train)

print("Non-zero coefficients:", np.sum(model.coef_ != 0), "out of 50")
print("Zeroed out features  :", np.sum(model.coef_ == 0), "out of 50")
```

---

### Step 5 — Auto-Tune with ElasticNetCV

```python
cv_model = ElasticNetCV(
    l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # try these mixes
    alphas   = np.logspace(-3, 1, 50),             # try 50 alpha values
    cv       = 5,                                  # 5-fold cross validation
    max_iter = 5000
)
cv_model.fit(X_train, y_train)

print("Best alpha    :", cv_model.alpha_)
print("Best l1_ratio :", cv_model.l1_ratio_)
```

> Never guess alpha manually. CV tries 50 × 5 = 250 fits and returns the best one — all using only training data. No data leakage.

**Why CV and not test set?**
```python
# ❌ WRONG — peeking at test data to pick alpha = data leakage
for alpha in [0.01, 0.1, 1.0]:
    score = model.score(X_test, y_test)  # cheating!

# ✅ RIGHT — CV uses only training data
cv_model = ElasticNetCV(cv=5)
cv_model.fit(X_train, y_train)           # never touches X_test
score = cv_model.score(X_test, y_test)  # honest evaluation
```

---

### Step 6 — Metrics

```python
y_pred = cv_model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.4f}")
```

---

### Step 7 — Coefficient Plot

```python
plt.figure(figsize=(12, 4))
plt.bar(range(len(cv_model.coef_)), cv_model.coef_, color='steelblue')
plt.axhline(0, color='red', lw=1, linestyle='--')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Elastic Net Coefficients — zeros = removed features")
plt.show()
```

> Bars at 0 = features the model discarded. Tall bars = features that truly drive predictions.

---

### Step 8 — Actual vs Predicted Plot

```python
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='cyan')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect')
plt.xlabel("Actual y")
plt.ylabel("Predicted ŷ")
plt.title(f"Actual vs Predicted  |  R² = {r2:.3f}")
plt.legend()
plt.show()
```

> Points close to the red diagonal = good predictions. Scatter away from it = errors.

---

### Step 9 — Simulate Correlated Features (to see l1_ratio < 1.0)

```python
# Default make_regression gives independent features → l1_ratio = 1.0 (Lasso wins)
# To see Elastic Net truly shine, add correlated features:

X_corr, y_corr = make_regression(
    n_samples=200, n_features=50,
    n_informative=10, noise=20,
    effective_rank=10,    # ← this adds correlations between features
    random_state=42
)

cv2 = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    alphas=np.logspace(-3, 1, 50), cv=5
).fit(StandardScaler().fit_transform(X_corr), y_corr)

print(f"l1_ratio: {cv2.l1_ratio_}")   # now < 1.0
print(f"alpha   : {cv2.alpha_:.4f}")
```

---

## 5. Results Interpretation

### Your Results

```
Best alpha    : 1.0481
Best l1_ratio : 1.0
MSE           : 481.85
RMSE          : 21.95
MAE           : 16.95
R²            : 0.9595
```

### What Each Means

| Result | Meaning |
|---|---|
| **l1_ratio = 1.0** | Pure Lasso won — your features were independent, no L2 needed. This is correct, not a mistake. |
| **alpha = 1.048** | Moderate regularization was enough to zero out irrelevant features |
| **R² = 0.9595** | Model explains 96% of variance in y — very good |
| **RMSE = 21.95** | Average prediction is off by ~22 units (in same units as y) |
| **MAE = 16.95** | Most predictions off by ~17 units — robust view of typical error |
| **MSE = 481.85** | Just RMSE² — don't interpret directly, used internally for math |

### Reading RMSE vs MAE Gap

```
RMSE = 21.95,  MAE = 16.95  →  gap ≈ 5

Gap is small   → errors are consistent, no badly wrong predictions
Gap is large   → a few predictions are very far off (outlier errors)
Your gap of 5 → acceptable, model is consistently good
```

### R² Scale

```
R² = 0.00  →  model is as dumb as predicting the mean every time
R² = 0.50  →  explains half the pattern
R² = 0.85  →  solid model
R² = 0.96  →  very good ✅  (your result)
R² = 1.00  →  perfect (suspicious — likely overfitting)
```

### Sanity Check Your RMSE

```python
# RMSE means nothing without context — compare to y range
rmse_pct = 21.95 / (y_test.max() - y_test.min()) * 100
print(f"RMSE is {rmse_pct:.1f}% of y range")
# Under 10% → great
# 10–20%    → acceptable
# Over 20%  → needs improvement

# Also compare to a dumb baseline
baseline = ((y_test - y_test.mean())**2).mean()**0.5
print(f"Baseline RMSE : {baseline:.2f}")
print(f"Your RMSE     : 21.95")
# Your model should crush the baseline
```

---

## 6. When to Use / Not Use

### ✅ Use Elastic Net When

- **Many features, few samples** (p > n) — e.g. 500 features, 200 rows. Regularization prevents overfitting.
- **Correlated features** — Lasso randomly picks one and drops the rest. Elastic Net groups them and keeps all with reduced weights.
- **You want feature selection** — coefficients go exactly to zero, irrelevant features automatically removed.
- **Unsure between Ridge or Lasso** — use Elastic Net, let CV decide via l1_ratio.
- **Genomics, text data, finance** — high dimensional data with lots of irrelevant or redundant features. Classic Elastic Net territory.

### ❌ Do NOT Use Elastic Net When

- **Few features (< 10–15)** — plain linear regression is fine. Regularization adds complexity without benefit.
- **All features are genuinely important** — zeroing them out hurts you. Use Ridge instead.
- **Non-linear relationships** — Elastic Net is still linear. Use XGBoost or Random Forest for curves and interactions.
- **Very large datasets (millions of rows)** — coordinate descent gets slow. Use `SGDRegressor(penalty='elasticnet')` instead.
- **You need probability outputs** — use logistic regression or classification models.

### 🆚 Which Model to Pick

```
Features correlated + want selection   →  Elastic Net ✅
Features independent + want selection  →  Lasso
Features correlated + keep all         →  Ridge
Few features, clean data               →  Linear Regression
Non-linear patterns                    →  XGBoost / Random Forest
Millions of rows                       →  SGDRegressor(penalty='elasticnet')
```

> **One Rule:** If you're doing linear regression with more than 20 features — always try Elastic Net with CV first. It costs nothing and often beats plain linear regression automatically.

---

## 7. Key Rules & Mental Model

### Rules

- Always **StandardScaler** before fitting — unscaled features get unfair penalties
- Always use **ElasticNetCV** not ElasticNet — never guess alpha, CV finds it safely
- **l1_ratio = 1.0 is not wrong** — it means Lasso was the right tool for that data
- Correlated features → l1_ratio drops below 1.0
- RMSE and R² are your two most important metrics
- MSE is just RMSE² — never interpret it directly

### Mental Model

```
Alpha ↑        →  stronger penalty  →  more zeros   →  simpler model
l1_ratio → 1   →  more like Lasso   →  aggressive zeroing
l1_ratio → 0   →  more like Ridge   →  smooth shrinkage, no zeros
CV             →  finds best alpha automatically, zero data leakage
R²   → 1.0     →  perfect fit
RMSE           →  your error in real units (lower = better)
RMSE vs MAE gap →  small gap = consistent errors, large gap = outlier errors
```

### l1_ratio Cheatsheet

```python
# l1_ratio = 0.0  →  pure Ridge  (no zeros, smooth shrinkage)
# l1_ratio = 0.5  →  Elastic Net (some zeros, balanced)
# l1_ratio = 1.0  →  pure Lasso  (aggressive zeroing)

# Many correlated features?    →  l1_ratio around 0.5
# Want aggressive selection?   →  l1_ratio closer to 1.0
# Features mostly independent? →  Lasso wins (l1_ratio = 1.0)
```

---

*Built while learning Elastic Net from scratch — code, results, and intuition all in one place.*