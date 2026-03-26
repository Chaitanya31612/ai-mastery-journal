# Project 01 — Linear Regression from Scratch

> **What you'll build:** A linear regression model using gradient descent, without any ML library.
> **Why it matters:** This is the foundation — every model you'll ever use is a fancier version of this.

---

## 🎯 Objective

Build a model that predicts house prices from area, using only NumPy.

By the end, you'll understand:
- How models *actually* learn (gradient descent)
- What loss functions are and why they matter
- How learning rate affects training
- What R² score means for regression

---

## 🧠 Concepts Used

| Concept | Where It's Used |
|---------|----------------|
| Dot product | Making predictions: `y = X · W + b` |
| Mean Squared Error | Measuring how wrong predictions are |
| Derivatives | Computing gradients (direction to improve) |
| Gradient Descent | The learning loop that finds optimal weights |

---

## 📁 Code Location

```
code/from-scratch/linear_regression_gd.py
```

---

## 🏃 How to Run

```bash
cd machine-learning/code
source venv/bin/activate    # Activate the virtual environment
cd from-scratch
python linear_regression_gd.py
```

This will:
1. Generate synthetic house price data
2. Train the model using gradient descent
3. Print learned weights vs true weights
4. Save two plots:
   - `linear_regression_results.png` — data, fit line, loss curve, predicted vs actual
   - `learning_rate_experiment.png` — how learning rate affects convergence
5. Compare results with scikit-learn (sanity check)

---

## 📊 Expected Output

```
🏗️  LINEAR REGRESSION FROM SCRATCH
======================================================
📊 Dataset: 100 samples, 1 feature(s)
   True relationship: y = 3x + 5 (with noise)
   Train: 80 samples | Test: 20 samples

🎓 Training with gradient descent...
  Step    0 | Loss: 40.xxxx | Weight: x.xxxx | Bias: x.xxxx
  ...
  ✅ Training complete! Final loss: ~1.0

📈 Results:
   Learned weight: ~3.0 (true: 3.0)
   Learned bias:   ~5.0 (true: 5.0)
   Train R²:       ~0.90
   Test R²:        ~0.85
```

---

## 🔬 Experiments to Try

### 1. Change the noise level
```python
X, y = generate_data(noise=0)   # No noise → R² should be ~1.0
X, y = generate_data(noise=50)  # High noise → R² drops significantly
```
**Learning:** Noise = irreducible error. No model can eliminate it.

### 2. Change the learning rate
```python
model = LinearRegressionFromScratch(learning_rate=0.001)  # Very slow convergence
model = LinearRegressionFromScratch(learning_rate=1.5)    # Diverges!
```
**Learning:** Learning rate is the single most impactful hyperparameter.

### 3. More training steps
```python
model = LinearRegressionFromScratch(n_iterations=50)     # Underfitting
model = LinearRegressionFromScratch(n_iterations=10000)  # Diminishing returns
```
**Learning:** There's a sweet spot — the loss curve flattens out.

---

## ✅ Confidence Check

After completing this project, you should be able to answer:

- [ ] What is gradient descent doing at each step?
- [ ] Why do we subtract the gradient (and not add it)?
- [ ] What happens if the learning rate is too large?
- [ ] What does `R² = 0` mean?
- [ ] Why did we split into train/test?
- [ ] Why does our model match scikit-learn's output?

---

## 🔗 Related Docs

- [02-math-for-ml-practical.md](../general/02-math-for-ml-practical.md) — The math behind this implementation
- [03-training-validation-testing.md](../general/03-training-validation-testing.md) — Why we split data
- [01-ml-overview-and-lifecycle.md](../general/01-ml-overview-and-lifecycle.md) — Where this fits in the ML lifecycle

---

*Next project → `02-logistic-regression-from-scratch.md` (same idea, but for classification)*
