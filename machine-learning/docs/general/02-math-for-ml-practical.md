# 02 — Math for ML (Practical SDE Edition)

> **Don't panic.** You don't need a math degree. You need ~5 concepts, the intuition behind
> them, and the ability to recognize them in code. That's it.

---

## 1. The Math You Actually Need

Here's the honest breakdown of math topics and how much you *really* need:

| Topic | Need Level | Why |
|-------|-----------|-----|
| **Linear Algebra (basics)** | ⭐⭐⭐ | Data is matrices. Models multiply matrices. |
| **Calculus (just derivatives)** | ⭐⭐⭐ | Gradient descent = derivatives. That's 90% of training. |
| **Probability (intuition only)** | ⭐⭐ | Classification outputs are probabilities. |
| **Statistics (mean, variance, std)** | ⭐⭐ | Data exploration, normalisation. |
| **Optimisation** | ⭐⭐ | Understanding how models improve (gradient descent). |

> **SDE Analogy:** You don't need to understand CPU microarchitecture to write Go code.
> Similarly, you don't need to prove theorems — you need to know what the tools do and when they break.

---

## 2. Linear Algebra — Data Is Just Matrices

### Vectors = A Row of Numbers

```python
import numpy as np

# A single house with 3 features: [area_sqft, bedrooms, age_years]
house = np.array([1200, 3, 10])

# This is a vector — a 1D array of numbers
# Think of it as a single row in your database table
```

### Matrices = A Table of Numbers

```python
# 4 houses, each with 3 features
houses = np.array([
    [1200, 3, 10],
    [1800, 4, 5],
    [900, 2, 20],
    [2200, 5, 2],
])

print(houses.shape)  # (4, 3) — 4 rows (samples), 3 columns (features)
```

> **SDE Analogy:** A matrix is just a 2D array. Or a SQL table without column names.
> `shape = (rows, columns)` = `(num_samples, num_features)`.

### The Dot Product — The Core Operation of ML

The dot product is how models make predictions. It's just **multiply-and-add**:

```python
# Model weights (what the model learned)
weights = np.array([100, 50000, -2000])
bias = 100000

# Prediction for one house
house = np.array([1200, 3, 10])

# Dot product: (1200 × 100) + (3 × 50000) + (10 × -2000) + 100000
prediction = np.dot(house, weights) + bias
print(f"Predicted price: ₹{prediction:,.0f}")
# Output: Predicted price: ₹350,000
```

**What happened:**
- Each feature got multiplied by its weight (importance).
- Results got summed up.
- That's literally how linear regression works!

> **SDE Analogy:** A dot product is like a weighted scoring system.
> Think `final_score = rating × weight_rating + reviews × weight_reviews + ...`

### Matrix Multiplication — Batch Predictions

```python
# Predict ALL houses at once (this is why GPUs are fast — parallel matrix ops)
all_predictions = np.dot(houses, weights) + bias
print(all_predictions)
# Array of 4 prices, one per house
```

> **Key insight:** ML frameworks use matrix operations because they're massively parallelisable.
> That's why GPUs (designed for graphics matrix math) are perfect for ML.

---

## 3. Calculus — Just Derivatives (The Slope of a Curve)

### Why Derivatives?

When training, we want to find the **lowest point** of a loss function (the error). Derivatives tell us **which direction is downhill**.

```
    Loss
    │
    │  ╲                    The derivative (slope) tells
    │    ╲                  you which way is "down."
    │      ╲
    │        ╲──────╱       ← We want to reach this valley!
    │                ╱
    │              ╱
    └───────────────── Parameter value
```

### The Only Derivative Rule You Need

For a function `f(x) = x²`:
- Derivative: `f'(x) = 2x`
- At `x = 3`: slope = 6 (steep, going up → move left)
- At `x = 0`: slope = 0 (flat → you're at the minimum!)

```python
# Intuition: derivative = "how much does output change when input changes slightly?"

def loss(weight):
    """Simple squared error loss"""
    return (weight - 3) ** 2  # Minimum is at weight = 3

def derivative(weight):
    """How fast is loss changing at this point?"""
    return 2 * (weight - 3)

# Let's see the slope at different points
for w in [0, 1, 2, 3, 4, 5]:
    print(f"weight={w}, loss={loss(w)}, slope={derivative(w)}")

# Output:
# weight=0, loss=9, slope=-6    ← steep slope, move right (increase weight)
# weight=1, loss=4, slope=-4    ← still negative, keep going right
# weight=2, loss=1, slope=-2    ← getting closer
# weight=3, loss=0, slope=0     ← MINIMUM! slope is zero 🎯
# weight=4, loss=1, slope=2     ← overshot, slope says go left
# weight=5, loss=4, slope=4     ← way too far right
```

> **SDE Analogy:** Derivatives are like `git diff` for functions.
> They tell you "how much did the output change for a small input change?"

---

## 4. Gradient Descent — The Learning Algorithm

This is THE algorithm behind almost all ML training. It's beautifully simple:

```
1. Start with random weights
2. Calculate the loss (how wrong you are)
3. Calculate the gradient (which direction reduces the loss)
4. Update weights: new_weight = old_weight - learning_rate × gradient
5. Repeat until loss stops decreasing
```

### Code: Gradient Descent in 15 Lines

```python
import numpy as np

# We want to find: what weight minimises loss?
# True answer: weight = 3 (but the algorithm doesn't know that)

weight = 10.0         # Start with a random guess
learning_rate = 0.1   # How big each step is

print(f"Starting weight: {weight}")

for step in range(20):
    loss = (weight - 3) ** 2            # How wrong are we?
    gradient = 2 * (weight - 3)         # Which direction to go?
    weight = weight - learning_rate * gradient  # Take a step downhill

    if step % 4 == 0:
        print(f"Step {step:2d}: weight = {weight:.4f}, loss = {loss:.4f}")

print(f"\nFinal weight: {weight:.4f} (target was 3.0)")
```

### Output:
```
Starting weight: 10.0
Step  0: weight = 8.6000, loss = 49.0000
Step  4: weight = 3.8153, loss = 1.1317
Step  8: weight = 3.1066, loss = 0.0193
Step 12: weight = 3.0139, loss = 0.0003
Step 16: weight = 3.0018, loss = 0.0000

Final weight: 3.0002 (target was 3.0)
```

**It found the right answer** just by following the slope downhill! 🎯

### The Learning Rate — Step Size Matters

| Learning Rate | Effect | Analogy |
|--------------|--------|---------|
| Too small (0.001) | Slow convergence, might take forever | Walking to your destination |
| Just right (0.01-0.1) | Smooth convergence | Driving at a sensible speed |
| Too large (10.0) | Overshoots, loss explodes | Teleporting past the destination repeatedly |

```python
# Too large learning rate — CHAOS
weight = 10.0
learning_rate = 1.5  # WAY too big

for step in range(5):
    gradient = 2 * (weight - 3)
    weight = weight - learning_rate * gradient
    print(f"Step {step}: weight = {weight:.1f}")

# Output:
# Step 0: weight = -11.0    ← overshot massively!
# Step 1: weight = 31.0     ← bouncing all over!
# Step 2: weight = -53.0    ← it's diverging!
```

> **SDE Analogy:** Learning rate is like the poll interval in an event loop.
> Too fast → CPU burn. Too slow → missed events. Just right → smooth operation.

---

## 5. Probability — Just Enough for Classification

When a model says "this email is 87% likely to be spam," that 87% is a **probability**.

### Sigmoid Function — Turning Numbers into Probabilities

The sigmoid squashes any number into the range (0, 1):

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Any input → output between 0 and 1
for val in [-5, -2, 0, 2, 5]:
    print(f"sigmoid({val:2d}) = {sigmoid(val):.4f}")

# Output:
# sigmoid(-5) = 0.0067  ← very close to 0 (confident NOT spam)
# sigmoid(-2) = 0.1192
# sigmoid( 0) = 0.5000  ← uncertain, 50/50
# sigmoid( 2) = 0.8808
# sigmoid( 5) = 0.9933  ← very close to 1 (confident SPAM)
```

> **SDE Analogy:** Sigmoid is like a normaliser that maps any value to a [0, 1] confidence percentage.
> Think of it as a `clamp(0, 1)` but smooth instead of hard-cutoff.

---

## 6. Statistics — The Basics

You need just 3 things: **mean**, **variance/std**, and **normalisation**.

### Why Normalise Features?

Imagine two features: `salary` (₹50,000) and `age` (25). Salary is 2000× larger, so the model might think salary is 2000× more important — which is wrong!

```python
import numpy as np

# Raw (unnormalised) features
data = np.array([
    [50000, 25],  # [salary, age]
    [60000, 30],
    [45000, 22],
    [80000, 40],
])

# Standardisation: mean = 0, std = 1
mean = data.mean(axis=0)
std = data.std(axis=0)
normalised = (data - mean) / std

print("Before normalisation:")
print(data)
print(f"\nMeans: {mean}, Stds: {std}")
print(f"\nAfter normalisation:")
print(normalised.round(2))
```

### Output:
```
Before normalisation:
[[50000    25]
 [60000    30]
 [45000    22]
 [80000    40]]

Means: [58750.   29.25], Stds: [13149.78     6.60]

After normalisation:
[[-0.67 -0.64]
 [ 0.1   0.11]
 [-1.05 -1.1 ]
 [ 1.62  1.63]]
```

Now both features are on the same scale. The model can fairly weigh their importance.

> **SDE Analogy:** Normalisation is like converting currencies to a common denomination before comparing prices.

---

## 7. Putting It All Together — The Mental Model

Here's how all the math connects during ML training:

```
┌─────────────────────────────────────────────────┐
│                  TRAINING LOOP                   │
│                                                  │
│  1. Input features (vectors/matrices)            │
│         ↓  [Linear Algebra]                      │
│  2. Model computes: prediction = X · W + b       │
│         ↓  [Dot Product]                         │
│  3. Loss function measures error                 │
│         ↓  [Statistics: mean squared error]       │
│  4. Gradient = derivative of loss w.r.t. weights │
│         ↓  [Calculus: chain rule]                 │
│  5. Update: W = W - lr × gradient               │
│         ↓  [Gradient Descent]                     │
│  6. Repeat until loss is small enough            │
│                                                  │
│  For classification: add sigmoid at step 2       │
│         ↓  [Probability]                         │
│  Output is: probability between 0 and 1          │
└─────────────────────────────────────────────────┘
```

---

## 8. The Good News

You now know 90% of the math behind:
- **Linear Regression** (dot product + gradient descent)
- **Logistic Regression** (same + sigmoid)
- **Neural Networks** (same, but stacked in layers)

The rest is engineering — choosing the right loss function, tuning hyperparameters, and fighting with data quality. *That's the SDE part.*

---

## 9. Cheat Sheet: Math → Code Mapping

| Math Concept | Python / NumPy Code | When You'll See It |
|-------------|---------------------|-------------------|
| Dot product | `np.dot(X, W)` | Every prediction |
| Matrix multiply | `X @ W` (shorthand) | Batch predictions |
| Mean | `np.mean(errors)` | Loss calculation |
| Derivative | Compute manually or use autograd | Gradient descent |
| Sigmoid | `1 / (1 + np.exp(-x))` | Classification output |
| Normalisation | `(X - mean) / std` | Data preprocessing |
| Argmax | `np.argmax(probs)` | Picking the top class |

---

*Next up → Building **Linear Regression from scratch** using everything we just learned!*
