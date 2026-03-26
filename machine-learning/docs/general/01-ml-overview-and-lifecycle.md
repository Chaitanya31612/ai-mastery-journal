# 01 — ML Overview & Lifecycle

> **SDE Analogy:** ML is *compiler-driven programming* — instead of writing rules by hand,
> you hand the compiler (training) a bunch of examples and it *learns* the rules for you.

---

## 1. What IS Machine Learning, Really?

Imagine you're building a spam filter.

**Traditional programming:**
```
IF email contains "lottery" AND sender not in contacts → SPAM
```
You write every rule. Thousands of them. And spammers just change one word.

**Machine learning:**
```
Here are 50,000 emails labelled SPAM or NOT_SPAM.
Figure out the pattern yourself.
```
You give **data + answers** → the algorithm writes the rules.

### The One-Liner
> ML = algorithms that improve at a task by learning from data, instead of being explicitly programmed.

---

## 2. The Three Flavours of ML

| Type | What You Give It | What It Does | Real Example |
|------|-----------------|--------------|--------------|
| **Supervised** | Data + correct answers (labels) | Learns to predict the answer for new data | Spam filter, house price prediction |
| **Unsupervised** | Data only, no labels | Finds hidden structure/groups | Customer segmentation, anomaly detection |
| **Reinforcement** | Environment + reward signal | Learns by trial-and-error to maximise reward | Game-playing AI, self-driving cars |

> **SDE Analogy:**
> - Supervised = You have unit tests (labels). The model trains until tests pass.
> - Unsupervised = You have logs but no tests. The model finds clusters/patterns in your logs.
> - Reinforcement = A/B testing on steroids — the model keeps deploying, measures results, and improves.

For this entire ML track, we'll focus 90% on **Supervised Learning** — it's the workhorse of real-world ML.

---

## 3. Regression vs Classification

These are the two main supervised learning tasks:

| Task | Output | Example |
|------|--------|---------|
| **Regression** | A continuous number | "This house costs ₹85,00,000" |
| **Classification** | A category/label | "This email is SPAM" |

> **SDE Analogy:**
> - Regression = a function that returns a `float`
> - Classification = a function that returns an `enum`

**Quick sanity check:** If someone asks "predict how many users will churn," that sounds like a number, but it's actually classification — each user either churns or doesn't (yes/no). The *count* is just summing the predictions.

---

## 4. The ML Lifecycle (End-to-End)

Think of it like the **Software Development Lifecycle** you already know, but for models:

```
┌─────────────────────────────────────────────────────────┐
│                    ML LIFECYCLE                          │
│                                                         │
│  1. Problem Definition     ← "What are we predicting?"  │
│         ↓                                               │
│  2. Data Collection        ← Databases, APIs, scraping  │
│         ↓                                               │
│  3. Data Cleaning          ← Handle nulls, outliers     │
│         ↓                                               │
│  4. Feature Engineering    ← Transform raw → useful     │
│         ↓                                               │
│  5. Model Selection        ← Pick algorithm(s)          │
│         ↓                                               │
│  6. Training               ← Model learns from data     │
│         ↓                                               │
│  7. Evaluation             ← How good is it, really?    │
│         ↓                                               │
│  8. Tuning & Iteration     ← Improve, repeat 5-7       │
│         ↓                                               │
│  9. Deployment             ← Serve predictions (API)    │
│         ↓                                               │
│  10. Monitoring            ← Detect model decay/drift   │
└─────────────────────────────────────────────────────────┘
```

### SDE Mapping

| ML Step | SDE Equivalent |
|---------|---------------|
| Problem Definition | Product requirements doc |
| Data Collection | Setting up data pipeline |
| Data Cleaning | Input validation & sanitisation |
| Feature Engineering | Data transformation layer |
| Model Selection | Choosing the right library/framework |
| Training | Build step / compilation |
| Evaluation | Running test suite |
| Tuning | Performance optimisation |
| Deployment | CI/CD + shipping to prod |
| Monitoring | Observability (Grafana, alerts) |

---

## 5. What's a "Model"?

A model is just a **mathematical function with adjustable knobs (parameters)**.

```python
# Simplest model: a straight line
def predict(x, weight, bias):
    return weight * x + bias
```

- **Training** = finding the best values for `weight` and `bias` so the function gives good predictions.
- **Parameters** = the knobs the algorithm tunes automatically during training.
- **Hyperparameters** = the settings YOU choose before training (learning rate, number of trees, etc.)

> **SDE Analogy:**
> - Parameters = variables computed at runtime.
> - Hyperparameters = config values you set in `settings.yaml`.

---

## 6. What Makes ML Hard? (Honest Answer)

| Challenge | Why It's Hard |
|-----------|--------------|
| **Data quality** | Garbage in → garbage out. 80% of ML work is cleaning data. |
| **Overfitting** | Model memorises training data instead of learning patterns. Like a student who memorises answers but can't solve new problems. |
| **Choosing metrics** | Accuracy sounds great but can be misleading (99% accuracy on a dataset where 99% is one class). |
| **Deployment** | Training a model in a notebook is easy. Serving it reliably at scale is software engineering. |

---

## 7. Key Vocabulary Cheat Sheet

| Term | Plain English |
|------|--------------|
| **Feature** | An input column (e.g., `age`, `salary`) — think "function parameter" |
| **Label / Target** | The answer you're predicting (e.g., `house_price`) |
| **Training set** | Data the model learns from |
| **Test set** | Data held back to evaluate how well the model generalises |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | A chunk of data processed in one step (like pagination) |
| **Inference** | Using a trained model to make predictions on new data |
| **Pipeline** | A chain of data transforms + model, like a CI/CD pipeline for data |

---

## 8. Quick Code: Your First Model in 10 Lines

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load a classic dataset
X, y = load_iris(return_X_y=True)

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)          # ← This is where learning happens

# Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
# Output: Accuracy: 100.00%  (Iris is a toy dataset, real life won't be this clean!)
```

**What just happened:**
1. Loaded 150 flower measurements with 3 species labels.
2. Split into train/test (so we can honestly evaluate).
3. Decision tree learned rules from training data.
4. Tested on unseen data → got accuracy.

---

## 9. What's Next?

Now that you see the big picture, we'll dive into:
- **03-training-validation-testing.md** — How to split data properly (it's trickier than you think).
- **02-math-for-ml-practical.md** — The actual math you need (spoiler: it's less than you fear).
- Then we'll build **linear regression from scratch** to see how training actually works under the hood.

---

*Phase 0 ✅ — You now have a mental map of ML. Everything else is filling in details.*
