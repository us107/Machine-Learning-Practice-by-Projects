# Lasso Regression (L1 Regularization)

## What it is
**Lasso Regression** is linear regression with an **L1 penalty** that forces some coefficients to become **exactly zero**.  
This makes the model **simple, interpretable, and sparse**.

---

## Why it is used
Traditional linear regression:
- Uses all features
- Overfits when features are many
- Is hard to interpret

Lasso fixes this by **removing unimportant features automatically**.

---

## How it works (core idea)

Lasso minimizes:

\[
\text{Loss} = \sum (y - \hat{y})^2 + \lambda \sum |\beta|
\]

- First term → prediction error  
- Second term → penalty on coefficient size  
- **λ (lambda)** controls how strong the penalty is

---

## What L1 regularization does

- Penalizes every non-zero coefficient
- Small coefficients are **pushed to zero**
- Results in **feature selection**

> If a feature doesn’t contribute enough, its weight becomes zero.

---

## Effect of λ (lambda)

- λ = 0 → Normal Linear Regression
- Small λ → Few features removed
- Large λ → Many coefficients become zero
- Very large λ → Almost all features removed

---

## Key benefits

- Reduces overfitting
- Improves generalization
- Produces simpler models
- Makes models easier to explain

---

## When to use Lasso

- Many features, few are important
- High-dimensional data
- Need feature selection + prediction
- Interpretability matters

---

## When NOT to use Lasso

- Features are highly correlated
- All features are important
- Very small datasets

(In these cases, Ridge or Elastic Net works better.)

---

## Real-world use cases

- Credit risk prediction
- Customer churn analysis
- Gene selection in healthcare
- System performance modeling

---

## One-line intuition

**Lasso keeps only the features that truly matter and discards the rest.**
