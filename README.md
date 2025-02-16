# Bank Marketing Machine Learning Analysis

This report examines a dataset from direct marketing campaigns (phone calls) run by a Portuguese bank.

The aim is to understand what drives successful telemarketing efforts and identify key factors that can improve marketing strategies. The goal is to predict which customers are most likely to sign up for term deposit products through telemarketing. With better targeting, the bank can cut marketing costs and boost return on investment (ROI).

- **Model Type:** Classification model
- **Predictive Target:** Signed up for a term deposit (1) or not (0)


## Data Source  
This project uses the **UCI Machine Learning Repository: Bank Marketing Data Set (2012)**. You can access it here:  
[UCI Bank Marketing Dataset](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing) *(Accessed: August 26, 2022)*.

**Original Source:**  
Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, Elsevier, 62, 22-31. doi:10.1016/j.dss.2014.06.001

---

## Training and Testing Data
- **Total Records:** 41,188 (20 attributes)
- **Training Set:** 80% (32,950 records)
- **Testing Set:** 20% (8,238 records)

**Stratified Splitting:** Ensures balanced distribution of the target variable across sets.

---

## Data Preparation and Splitting (Python)
```python
# Import libraries
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
df = pd.read_csv('bank-additional-full.csv', sep=';')

# Encode target ('y'): 'yes' → 1, 'no' → 0
df['y'] = df['y'].map({'yes': 1, 'no': 0}).astype(int)

# Split into features and target
X = df.drop(columns='y')
y = df['y']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)
```

---
**Key Points:**
- **Stratification** maintains class balance.
- `random_state=7` ensures reproducibility.
- `map()` simplifies target encoding.

This setup ensures a strong foundation for modeling and evaluation.

## Evaluation Metrics
The dataset has an **11.26% positive class rate**, highlighting the importance of using multiple metrics beyond accuracy to evaluate model performance effectively.

### 1. **Accuracy**  
- **Definition:** The ratio of correctly predicted outcomes to total predictions.
- **Strength:** Simple and easy to interpret.
- **Limitation:** Misleading on imbalanced datasets.

### 2. **Recall (Sensitivity)**  
- **Definition:** The proportion of actual positives correctly identified.
- **Strength:** Essential for minimizing false negatives (e.g., identifying potential subscribers).
- **Limitation:** Only evaluates one performance aspect.

### 3. **Precision**  
- **Definition:** The proportion of predicted positives that are truly positive.
- **Strength:** Useful for minimizing false positives, ensuring correct customer targeting.
- **Limitation:** Focuses solely on one performance dimension.

### 4. **F1 Score**  
- **Definition:** The harmonic mean of precision and recall.
- **Strength:** Balances false positives and false negatives, suitable for imbalanced datasets.
- **Limitation:** Provides a single score, masking individual performance aspects.




