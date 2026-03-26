# 03 — Training, Validation & Testing

> **SDE Analogy:** Train/Val/Test is the ML version of Dev/Staging/Production environments.
> You'd never test code only in dev and ship straight to prod. Same idea.

---

## 1. Why Split Data at All?

Imagine you're a student. If the exam is *exactly the same questions* you practised on, a 100% score means nothing — you might have just memorised answers.

ML models have the same problem. If we evaluate on the *same data* we trained on, the model can just **memorise** instead of **learn**. That's called **overfitting**.

**The fix:** Hold back some data the model has *never seen* and test on that.

---

## 2. The Three Splits

```
┌─────────────────────────────────────────────────┐
│               Your Full Dataset                  │
│                                                  │
│  ┌──────────────┐  ┌─────────┐  ┌────────────┐  │
│  │  TRAINING    │  │  VAL    │  │   TEST     │  │
│  │  (60-80%)    │  │ (10-20%)│  │  (10-20%)  │  │
│  │              │  │         │  │            │  │
│  │  Model       │  │ Tune    │  │ Final      │  │
│  │  learns here │  │ here    │  │ exam here  │  │
│  └──────────────┘  └─────────┘  └────────────┘  │
└─────────────────────────────────────────────────┘
```

| Split | Purpose | When Used | SDE Analogy |
|-------|---------|-----------|-------------|
| **Training** | Model learns patterns from this data | During `model.fit()` | Dev environment — write & iterate code |
| **Validation** | Tune hyperparameters, pick best model | After training, before shipping | Staging — run integration tests, QA |
| **Test** | Final honest evaluation — touch ONCE | Very end, before deploy | Production smoke test — if this fails, don't ship |

---

## 3. The Golden Rule

> **Never let your model see test data during training or tuning.**

This is like not peeking at the exam paper before the exam. If you tune your model based on test results, you're *fitting to the test set* — your reported performance is now a lie.

> **SDE Analogy:** It's like writing unit tests *after* seeing the bugs. Your tests pass, but your code isn't actually robust.

---

## 4. Common Split Ratios

| Scenario | Train | Val | Test |
|----------|-------|-----|------|
| Large dataset (>100K rows) | 80% | 10% | 10% |
| Medium dataset (10K-100K) | 70% | 15% | 15% |
| Small dataset (<10K) | Use cross-validation instead | — | 20% |

---

## 5. Cross-Validation (The Smart Way)

When you don't have enough data for a proper 3-way split, use **k-fold cross-validation**:

```
Dataset split into 5 folds:

Round 1: [VAL] [Train] [Train] [Train] [Train]  → Score: 0.85
Round 2: [Train] [VAL] [Train] [Train] [Train]  → Score: 0.87
Round 3: [Train] [Train] [VAL] [Train] [Train]  → Score: 0.83
Round 4: [Train] [Train] [Train] [VAL] [Train]  → Score: 0.86
Round 5: [Train] [Train] [Train] [Train] [VAL]  → Score: 0.84

Average Score: 0.85 ± 0.015
```

Each data point gets to be in the validation set exactly once. This gives you a much more robust estimate of performance.

> **SDE Analogy:** It's like running your test suite on 5 different configurations of your system. If it passes all 5, you're much more confident it works.

---

## 6. Data Leakage — The Silent Killer

**Data leakage** = when information from the future (or from the test set) sneaks into your training process.

### Common Ways You Accidentally Leak Data

| Leak Type | Example | Why It's Bad |
|-----------|---------|--------------|
| **Target leakage** | Using `was_admitted_to_hospital` to predict `will_get_sick` | The effect is included as a cause |
| **Train-test contamination** | Normalising *all* data before splitting | Statistics from test data influence training |
| **Temporal leakage** | Randomly splitting time-series data | Future data leaks into training |

### The Fix: Always Split First, Then Transform

```python
# ❌ WRONG — leaks test info into training
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # Computed on FULL dataset
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT — transform after splitting
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Learn from train only
X_test_scaled = scaler.transform(X_test)          # Apply same transform
```

> **SDE Analogy:** Data leakage is like testing your API with a mocked database that already contains the expected outputs. Your tests pass, but the API doesn't actually work.

---

## 7. Practical Code: Proper Splitting

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
X, y = load_iris(return_X_y=True)

# --- METHOD 1: Simple hold-out split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # Reproducibility — like a seed
    stratify=y           # Keep class ratios balanced in both sets
)

# Scale AFTER splitting (no leakage!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Hold-out test accuracy: {test_score:.2%}")


# --- METHOD 2: Cross-validation (more robust) ---
model = DecisionTreeClassifier(random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"\n5-Fold CV scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
```

### Output:
```
Hold-out test accuracy: 100.00%

5-Fold CV scores: [0.96666667 0.96666667 0.9        0.93333333 1.        ]
Mean: 95.33% ± 3.44%
```

Notice how cross-validation gives a more nuanced picture (95.33%) compared to the lucky hold-out split (100%). That's the point!

---

## 8. Stratified Splitting — Why It Matters

If your dataset has **imbalanced classes** (e.g., 95% not-fraud, 5% fraud), a random split might give you a test set with *zero* fraud cases.

**Stratified split** ensures each split has roughly the same class proportions as the original data.

```python
# Without stratify — might get unlucky class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# With stratify — guarantees balanced proportions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
```

> **SDE Analogy:** Stratified splitting is like ensuring your A/B test has proportional user segments (free vs paid) in both groups.

---

## 9. Quick Reference: Decision Flowchart

```
"How should I split my data?"

Is your dataset large (>50K samples)?
  YES → 80/10/10 train/val/test split
  NO  → Use 5-fold or 10-fold cross-validation
         + hold out 20% as a final test set

Is it time-series data?
  YES → Split chronologically (no randomisation!)
  NO  → Random split with stratification

Are you comparing many models / tuning a lot?
  YES → You NEED a separate validation set or CV
  NO  → Simple train/test might be enough for a quick check
```

---

## 10. Key Takeaways

1. **Always split before any data transformation** — prevents leakage.
2. **Validation set ≠ test set** — validation is for iteration, test is for final check.
3. **Cross-validation** gives more reliable estimates than a single split.
4. **Stratify** when classes are imbalanced.
5. **Random state** for reproducibility — your future self will thank you.

---

*Next up → `02-math-for-ml-practical.md` where we cover the actual math behind training (it's simpler than you think).*
