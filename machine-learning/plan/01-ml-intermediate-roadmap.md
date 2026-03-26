# ML Intermediate Roadmap (SDE Track)

## 1) Goal

Build practical, intermediate-level confidence in machine learning so AI/ML/DL discussions are concrete, not abstract.

By the end of this roadmap, you should be able to:

- Explain the full ML lifecycle end-to-end.
- Implement core algorithms from scratch (without framework magic).
- Build, evaluate, and improve real models using `scikit-learn`.
- Reason about bias/variance, overfitting, metrics, and model tradeoffs.
- Speak clearly about when ML works, why it fails, and what to do next.
- Transition smoothly into the `deep-learning/` track.

## 2) Timebox and Commitment

- Total duration: 16 weeks.
- Weekly effort: 8-12 hours.
- Session split:30% theory and notes in `docs/`

  60% implementation in `code/`

  10% recap, reflection, and communication practice

## 3) Folder Blueprint (Target State)

```text
machine-learning/
├── docs/
│   ├── general/
│   │   ├── 01-ml-overview-and-lifecycle.md
│   │   ├── 02-math-for-ml-practical.md
│   │   ├── 03-training-validation-testing.md
│   │   ├── 04-loss-functions-and-optimization.md
│   │   ├── 05-bias-variance-regularization.md
│   │   ├── 06-supervised-learning-regression.md
│   │   ├── 07-supervised-learning-classification.md
│   │   ├── 08-tree-models-and-ensembles.md
│   │   ├── 09-feature-engineering-and-pipelines.md
│   │   ├── 10-unsupervised-learning.md
│   │   ├── 11-model-interpretability-and-fairness.md
│   │   ├── 12-ml-system-design-basics.md
│   │   └── 13-bridge-to-deep-learning.md
│   └── projects/
│       ├── 01-linear-regression-from-scratch.md
│       ├── 02-logistic-regression-from-scratch.md
│       ├── 03-tabular-classification-pipeline.md
│       ├── 04-tree-and-ensemble-benchmark.md
│       ├── 05-customer-segmentation-kmeans.md
│       ├── 06-cat-dog-classical-vision-baseline.md
│       └── 07-ml-capstone.md
├── code/
│   ├── from-scratch/
│   ├── sklearn/
│   ├── projects/
│   ├── utils/
│   ├── datasets/
│   └── notebooks/
└── plan/
    └── 01-ml-intermediate-roadmap.md
```

## 4) Learning Phases (16 Weeks)

## Phase 0 (Week 1): Setup and Orientation

### Outcomes

- Local environment is stable and repeatable.
- You can load data, inspect data, and run first baseline models.

### Deliverables

- `docs/general/01-ml-overview-and-lifecycle.md`
- `docs/general/03-training-validation-testing.md`
- `code/notebooks/00-environment-and-first-model.ipynb`

### Topics

- ML problem framing: regression vs classification vs clustering.
- Dataset splits: train, validation, test.
- Baseline model concept and why it matters.

## Phase 1 (Weeks 2-4): Fundamentals You Actually Use

### Outcomes

- Strong intuition for the math used in daily ML work.
- Ability to debug training behavior instead of guessing.

### Deliverables

- `docs/general/02-math-for-ml-practical.md`
- `docs/general/04-loss-functions-and-optimization.md`
- `docs/general/05-bias-variance-regularization.md`
- `code/from-scratch/linear_regression_gd.py`
- `code/from-scratch/logistic_regression_gd.py`

### Topics

- Vectors, matrices, dot product, gradients.
- Cost/loss functions.
- Gradient descent and learning rate tuning.
- Overfitting, underfitting, regularization (`L1`, `L2`).

## Phase 2 (Weeks 5-7): Supervised ML in Practice

### Outcomes

- Build proper regression/classification pipelines with metrics that make sense.
- Compare multiple models, not just train one.

### Deliverables

- `docs/general/06-supervised-learning-regression.md`
- `docs/general/07-supervised-learning-classification.md`
- `docs/general/09-feature-engineering-and-pipelines.md`
- `docs/projects/03-tabular-classification-pipeline.md`
- `code/sklearn/regression_baselines.py`
- `code/sklearn/classification_baselines.py`
- `code/projects/tabular_classification_pipeline/`

### Topics

- Regression metrics: MAE, RMSE, R2.
- Classification metrics: precision, recall, F1, ROC-AUC, PR-AUC.
- Preprocessing with `Pipeline` and `ColumnTransformer`.
- Cross-validation and hyperparameter search basics.

## Phase 3 (Weeks 8-10): Trees, Ensembles, and Unsupervised Learning

### Outcomes

- Understand why tree-based models dominate tabular ML.
- Build and interpret clustering outputs.

### Deliverables

- `docs/general/08-tree-models-and-ensembles.md`
- `docs/general/10-unsupervised-learning.md`
- `docs/projects/04-tree-and-ensemble-benchmark.md`
- `docs/projects/05-customer-segmentation-kmeans.md`
- `code/projects/tree_ensemble_benchmark/`
- `code/projects/customer_segmentation/`

### Topics

- Decision trees, Random Forest, Gradient Boosting.
- Feature importance and model interpretation basics.
- K-Means clustering and cluster evaluation.

## Phase 4 (Weeks 11-13): Vision Project (Classical ML Baseline)

### Outcomes

- Build an image classification pipeline without jumping directly to deep learning.
- Understand feature extraction and why CNNs later replace manual features.

### Deliverables

- `docs/projects/06-cat-dog-classical-vision-baseline.md`
- `code/projects/cat_dog_classical_baseline/`

### Topics

- Image preprocessing.
- Feature extraction (HOG or color/texture features).
- Binary classification with SVM or logistic regression.
- Error analysis: confusion matrix and false-positive/false-negative cases.

## Phase 5 (Weeks 14-16): Capstone and Communication Confidence

### Outcomes

- Deliver one complete end-to-end ML project from data to report.
- Explain decisions confidently in technical discussions.

### Deliverables

- `docs/general/11-model-interpretability-and-fairness.md`
- `docs/general/12-ml-system-design-basics.md`
- `docs/general/13-bridge-to-deep-learning.md`
- `docs/projects/07-ml-capstone.md`
- `code/projects/ml_capstone/`
- `machine-learning/README.md` updated with project links

### Topics

- Model explainability basics (global + local explanations).
- Data leakage and production failure modes.
- Model monitoring ideas (drift, decay, retraining triggers).
- Clear communication of tradeoffs and limitations.

## 5) Weekly Execution Template

Repeat this every week:

1. Day 1: Read concept doc section and write your own explanation in `docs/`.
2. Day 2: Implement from scratch or baseline model.
3. Day 3: Add metrics, plots, and failure analysis.
4. Day 4: Refactor into clean modules in `code/projects/...`.
5. Day 5: Write "what I learned / what failed / next change" notes.

Minimum weekly output:

- 1 concept note update.
- 1 runnable script or notebook.
- 1 short project log with errors and fixes.

## 6) Project Sequence (In Recommended Order)

1. Linear Regression from Scratch
   - Learn gradients, loss curves, normalization effects.
2. Logistic Regression from Scratch
   - Learn sigmoid, decision threshold, precision-recall tradeoff.
3. Tabular Classification Pipeline
   - Learn preprocessing + model selection + validation flow.
4. Tree and Ensemble Benchmark
   - Learn model comparison and practical baseline setting.
5. Customer Segmentation (K-Means)
   - Learn unsupervised problem framing and interpretation.
6. Cat vs Dog (Classical Vision Baseline)
   - Learn image feature engineering and classifier limitations.
7. Capstone
   - Learn end-to-end ownership and communication.

## 7) Confidence Checklist (Intermediate Level)

You are at intermediate level when you can do all of this without notes:

- Explain train/validation/test and data leakage with examples.
- Pick metrics based on business risk, not habit.
- Diagnose overfitting and apply at least 3 fixes.
- Implement linear/logistic regression from scratch.
- Build an sklearn pipeline and compare 3+ candidate models.
- Explain confusion matrix tradeoffs for a real threshold decision.
- Describe why classical vision struggles and why deep learning helps.
- Present one capstone with model choice rationale and limitations.

## 8) Rules to Keep This Practical

- Always start with a baseline model.
- Never optimize before evaluating baseline errors.
- Track every experiment in a simple markdown log.
- Prefer fewer models with deeper analysis over many shallow runs.
- Treat failed experiments as required deliverables, not waste.

## 9) Exit Criteria to Start `deep-learning/`

Move to deep learning only after:

- You finish all 7 project docs in `docs/projects/`.
- You complete one capstone with reproducible code.
- You can explain gradient descent and regularization comfortably.
- You can discuss ML model failure modes in production-like settings.

Then start `deep-learning/` with:

- Neural network fundamentals (from scratch with numpy).
- PyTorch training loop internals.
- CNN-based cat vs dog classifier (compare against ML baseline here).

## 10) First 2 Weeks Action Plan (Start Immediately)

1. Create first two docs:
   - `docs/general/01-ml-overview-and-lifecycle.md`
   - `docs/general/02-math-for-ml-practical.md`
2. Implement:
   - `code/from-scratch/linear_regression_gd.py`
3. Add notebook:
   - `code/notebooks/01-linear-regression-experiments.ipynb`
4. Write one reflection note:
   - What clicked, what was confusing, what to revisit next week.

If these are done cleanly, continue to Phase 1 and keep the same weekly rhythm.
