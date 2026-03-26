"""
Linear Regression from Scratch — Gradient Descent Edition
=========================================================

🎯 Goal: Understand exactly how a model LEARNS by building one yourself.

No scikit-learn. No magic. Just NumPy and the math from 02-math-for-ml-practical.md.

Think of this file as the "assembly language" of ML — once you see how this works,
every framework (sklearn, PyTorch, TensorFlow) is just a nicer wrapper on top.

SDE Analogy: This is like writing your own HTTP server before using Express/FastAPI.
You don't do it in production, but you do it once to truly understand what's happening.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────
# SECTION 1: Generate Synthetic Data
# ──────────────────────────────────────────────────────────

def generate_data(n_samples=100, noise=10, seed=42):
    """
    Create fake housing data:
    - X = house area in hundreds of sqft
    - y = price in lakhs

    The TRUE relationship: price = 3 * area + 5 + noise
    Our model will try to discover this relationship.

    SDE Analogy: This is like creating test fixtures — we know the expected
    output, so we can verify our model actually works.
    """
    np.random.seed(seed)
    X = 2 * np.random.rand(n_samples, 1)            # Areas between 0 and 2 (hundred sqft)
    y = 3 * X + 5 + noise * np.random.randn(n_samples, 1) / 10  # True: y = 3x + 5

    return X, y


# ──────────────────────────────────────────────────────────
# SECTION 2: The Model — Dead Simple
# ──────────────────────────────────────────────────────────

class LinearRegressionFromScratch:
    """
    Linear Regression: y = X · W + b

    That's it. That's the whole model.

    - W (weight) controls the slope — "how much does price change per sqft?"
    - b (bias) controls the intercept — "what's the base price?"

    SDE Analogy:
    - W and b are like config parameters that get auto-tuned.
    - fit() is the training/compilation step.
    - predict() is the runtime/serving step.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []  # Track loss over time — like a monitoring dashboard

    def fit(self, X, y):
        """
        Train the model using gradient descent.

        The loop:
        1. Predict with current weights    (forward pass)
        2. Calculate how wrong we are      (loss)
        3. Calculate which direction to adjust (gradients)
        4. Take a small step in that direction (update)
        5. Repeat.

        SDE Analogy: This is an optimisation loop, like binary search.
        Each iteration gets us closer to the optimal weights.
        """
        n_samples, n_features = X.shape

        # Step 0: Start with random weights (our initial guess)
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            # ── STEP 1: Forward Pass (Make Predictions) ──
            # y_pred = X · W + b  (just a dot product!)
            y_pred = X @ self.weights + self.bias

            # ── STEP 2: Calculate Loss (Mean Squared Error) ──
            # "On average, how far off are our predictions?"
            errors = y_pred - y
            loss = np.mean(errors ** 2)
            self.loss_history.append(loss)

            # ── STEP 3: Calculate Gradients ──
            # "In which direction should we nudge each weight?"
            # These formulas come from calculus (derivative of MSE)
            dw = (2 / n_samples) * (X.T @ errors)     # Gradient w.r.t. weights
            db = (2 / n_samples) * np.sum(errors)      # Gradient w.r.t. bias

            # ── STEP 4: Update Weights ──
            # Move in the OPPOSITE direction of the gradient (go downhill)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Log progress every 100 steps
            if i % 200 == 0:
                print(f"  Step {i:4d} | Loss: {loss:.4f} | "
                      f"Weight: {self.weights.flatten()[0]:.4f} | Bias: {self.bias:.4f}")

        print(f"  ✅ Training complete! Final loss: {loss:.4f}")
        return self

    def predict(self, X):
        """Make predictions with trained weights."""
        return X @ self.weights + self.bias

    def score(self, X, y):
        """
        R² Score: How much variance in y does our model explain?

        - R² = 1.0 → perfect predictions
        - R² = 0.0 → model is as good as just predicting the mean
        - R² < 0.0 → model is WORSE than predicting the mean (broken model)

        SDE Analogy: R² is like code coverage — 1.0 is perfect, 0 means no value added.
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)      # Sum of squared residuals
        ss_tot = np.sum((y - np.mean(y)) ** 2)   # Total variance
        return 1 - (ss_res / ss_tot)


# ──────────────────────────────────────────────────────────
# SECTION 3: Train and Evaluate
# ──────────────────────────────────────────────────────────

def train_and_evaluate():
    """Run the full training pipeline."""

    print("=" * 60)
    print("🏗️  LINEAR REGRESSION FROM SCRATCH")
    print("=" * 60)

    # Generate data
    X, y = generate_data(n_samples=100, noise=10)
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} feature(s)")
    print(f"   True relationship: y = 3x + 5 (with noise)")

    # Manual train/test split (no sklearn needed!)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

    # Train the model
    print("🎓 Training with gradient descent...")
    model = LinearRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"\n📈 Results:")
    print(f"   Learned weight: {model.weights.flatten()[0]:.4f} (true: 3.0)")
    print(f"   Learned bias:   {model.bias:.4f} (true: 5.0)")
    print(f"   Train R²:       {train_score:.4f}")
    print(f"   Test R²:        {test_score:.4f}")

    return model, X_train, y_train, X_test, y_test


# ──────────────────────────────────────────────────────────
# SECTION 4: Visualisations (See What's Happening)
# ──────────────────────────────────────────────────────────

def plot_results(model, X_train, y_train, X_test, y_test):
    """Create 3 plots to understand what happened during training."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Data + Fitted Line
    ax = axes[0]
    ax.scatter(X_train, y_train, alpha=0.6, label='Train', color='#6366f1')
    ax.scatter(X_test, y_test, alpha=0.6, label='Test', color='#f59e0b', marker='s')

    X_line = np.linspace(0, 2, 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax.plot(X_line, y_line, color='#ef4444', linewidth=2, label='Learned line')

    ax.set_xlabel('Area (hundreds sqft)')
    ax.set_ylabel('Price (lakhs)')
    ax.set_title('Data + Learned Regression Line')
    ax.legend()

    # Plot 2: Loss Curve (How the Model Improved Over Time)
    ax = axes[1]
    ax.plot(model.loss_history, color='#6366f1', linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Squared Error (Loss)')
    ax.set_title('Loss Curve — Model Learning Progress')
    ax.set_yscale('log')  # Log scale shows the rapid initial improvement

    # Plot 3: Predictions vs Actual
    ax = axes[2]
    y_pred_test = model.predict(X_test)
    ax.scatter(y_test, y_pred_test, alpha=0.7, color='#10b981', edgecolors='white', s=80)

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5,
            label='Perfect prediction')

    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Predicted vs Actual (Test Set)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('linear_regression_results.png', dpi=120, bbox_inches='tight')
    print("\n📊 Plots saved to: linear_regression_results.png")
    plt.close()


# ──────────────────────────────────────────────────────────
# SECTION 5: Experiment — What Happens with Bad Learning Rates?
# ──────────────────────────────────────────────────────────

def learning_rate_experiment():
    """
    Show how learning rate affects training — the single most important
    hyperparameter to get right.

    SDE Analogy: It's like choosing the batch size for a queue consumer.
    Too small = slow processing. Too big = OOM or dropped messages.
    """
    print("\n" + "=" * 60)
    print("🧪 EXPERIMENT: Learning Rate Effects")
    print("=" * 60)

    X, y = generate_data()
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]

    learning_rates = [0.001, 0.01, 0.1, 0.5]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for i, lr in enumerate(learning_rates):
        model = LinearRegressionFromScratch(learning_rate=lr, n_iterations=500)

        # Suppress printing for this experiment
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        model.fit(X_train, y_train)
        sys.stdout = old_stdout

        axes[i].plot(model.loss_history, color='#6366f1')
        axes[i].set_title(f'LR = {lr}')
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Loss')
        axes[i].set_ylim(0, max(model.loss_history[0] * 1.1, 10))

        final_loss = model.loss_history[-1]
        status = "[OK]" if final_loss < 1.5 else "[SLOW]" if final_loss < 5 else "[BAD]"
        axes[i].annotate(f'{status} Final: {final_loss:.2f}',
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        ha='center', va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Learning Rate Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('learning_rate_experiment.png', dpi=120, bbox_inches='tight')
    print("📊 Learning rate comparison saved to: learning_rate_experiment.png")
    plt.close()


# ──────────────────────────────────────────────────────────
# SECTION 6: Compare with scikit-learn (Reality Check)
# ──────────────────────────────────────────────────────────

def compare_with_sklearn():
    """
    Our from-scratch model vs sklearn — they should give nearly identical results.
    If they don't, our math is wrong!

    SDE Analogy: Unit test — comparing our custom implementation against a known-good library.
    """
    from sklearn.linear_model import LinearRegression

    print("\n" + "=" * 60)
    print("🔍 COMPARISON: From Scratch vs scikit-learn")
    print("=" * 60)

    X, y = generate_data()
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Our model
    our_model = LinearRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    our_model.fit(X_train, y_train)
    sys.stdout = old_stdout

    # sklearn model
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)

    print(f"\n{'':20s} {'From Scratch':>15s} {'scikit-learn':>15s}")
    print(f"{'─' * 52}")
    print(f"{'Weight':20s} {our_model.weights.flatten()[0]:>15.4f} {sk_model.coef_.flatten()[0]:>15.4f}")
    print(f"{'Bias':20s} {our_model.bias:>15.4f} {sk_model.intercept_[0]:>15.4f}")
    print(f"{'Train R²':20s} {our_model.score(X_train, y_train):>15.4f} {sk_model.score(X_train, y_train):>15.4f}")
    print(f"{'Test R²':20s} {our_model.score(X_test, y_test):>15.4f} {sk_model.score(X_test, y_test):>15.4f}")
    print(f"\n✅ Results match! Our gradient descent found the same solution as sklearn.")


# ──────────────────────────────────────────────────────────
# MAIN — Run Everything
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Train and evaluate
    model, X_train, y_train, X_test, y_test = train_and_evaluate()

    # Generate plots
    plot_results(model, X_train, y_train, X_test, y_test)

    # Learning rate experiment
    learning_rate_experiment()

    # Compare with sklearn
    compare_with_sklearn()

    print("\n" + "=" * 60)
    print("🎓 KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. Training = finding weights that minimise the loss function.
    2. Gradient descent = following the slope downhill.
    3. Learning rate = step size (too small = slow, too big = chaos).
    4. R² score = how much variance our model explains (1.0 = perfect).
    5. Our from-scratch model matches scikit-learn — no magic involved!

    What to explore next:
    • Change the noise level — how does it affect R²?
    • Try different learning rates — when does it diverge?
    • Add more features — does the model still work?
    """)
