# Titanic Survival Classifier

Logistic regression built from scratch using only NumPy — no sklearn, no shortcuts.

**85.5% Accuracy | 0.79 F1**

---

## What I built

A binary classifier that predicts whether a Titanic passenger survived. Everything is manual — sigmoid, loss function, gradient descent, and evaluation metrics. The point was to actually understand how logistic regression works, not just call `.fit()`.

---

## Feature Engineering

The raw data needed some work before it was useful:

- **Age** — filled missing values using median per class + sex group, not a global median, because age distributions are different across those groups
- **Title** — extracted from the name (Mr, Mrs, Miss, etc.) since it captures gender, age, and social status better than the raw name field. Rare titles like Dr, Rev, Col got grouped together to reduce noise
- **Deck** — pulled the first letter of the Cabin column. Most cabins were missing so those became "Unknown", but the ones we have correlate with class and lifeboat proximity
- **FamilySize / IsAlone** — combined SibSp and Parch into one feature, then flagged solo travelers separately since traveling alone was a strong survival signal

---

## Implementation details

- **Weighted loss** — 62% of passengers didn't survive so the dataset is imbalanced. I added a survival weight in the cross-entropy loss so the model doesn't just learn to predict "not survived" for everyone
- **L2 regularization** — penalizes large weights to reduce overfitting. The bias is excluded because it just shifts the baseline, it's not part of the model's complexity
- **Threshold = 0.5** — After adding L2 regularization the model became confident enough that the default threshold works well, giving better accuracy and precision without sacrificing much recall
- **Normalization** — scaled using training set stats only, applied the same transform to test. Computing it on the full dataset would be leakage

---

## Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.8550 |
| Precision | 0.8222 |
| Recall    | 0.7708 |
| F1        | 0.7957 |

---

## Stack

Python, NumPy, Pandas, Matplotlib
