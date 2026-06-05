# Fairness in Automated Interview Screening

An investigation into algorithmic bias in automated hiring systems. Three classifiers are trained to predict hiring outcomes, with a focus on measuring racial fairness across demographic groups.

## Overview

Automated screening tools are increasingly used in recruitment, yet they risk encoding historical biases present in training data. This project examines how standard ML models behave on a hiring dataset when race is explicitly excluded as a training feature — and whether racial disparities persist through proxy variables.

## Dataset

50,000 synthetic candidate records (30k train / 10k val / 10k test), each with:

- **Demographic attributes**: Age, Sex, Race, Place of Birth
- **Background features**: Workclass, Education, Marital Status, Occupation, Relationship, Hours Per Week
- **Assessment scores**: Interview score, CV assessment score
- **Target**: `prior_hiring_decision` (binary)

Race is treated as the sensitive attribute and is dropped from all model inputs. The remaining features are used to predict hiring outcomes.

## Models

Each notebook trains a separate classifier using a consistent preprocessing pipeline (one-hot encoding for categoricals, standard scaling for numericals):

| Notebook | Model |
|---|---|
| `src/logistic_regression.ipynb` | Logistic Regression |
| `src/decision_tree.ipynb` | Decision Tree |
| `src/neural_network.ipynb` | MLP Neural Network |

## Fairness Analysis

Models are evaluated on both predictive accuracy and racial fairness metrics including:

- **Demographic Parity** — equal positive prediction rates across racial groups
- **Equal Opportunity** — equal true positive rates across groups
- **Equalized Odds** — equal TPR and FPR across groups

ROC curves and AUC scores complement the fairness analysis across all three models.

## Setup

```bash
pip install -r requirements.txt
```

Then open any notebook in `src/` and run cells in order. Data is pre-split and lives in `data/`.

## Report

`automated_interview_screening_report.pdf` contains the full written analysis and discussion of findings.
