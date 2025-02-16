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


## Model 2: Gradient Boosting Machine (GBM)

Gradient boosting is an approach where new decision trees are created that predict the residuals or errors of prior models and then combined to the final model. It is called gradient boosting because the models are fitted using any arbitrary differentiable loss function and uses a gradient descent algorithm to minimize the loss when adding new models.

###  Justification of model choice
- GBM can work for classification problems, in our context it can be used to predict if a client will subscribe to term deposit or not.
- GBM can handle linear and non-linear relationship
- Our dataset has a mix of numerical and categorical variables, XGBM can work well with both data types (after one hot encoded the categorical variable).
- It doesn’t require much assumption about the dataset and/or pre-processing

### Pre-processing performed
GBM doesn’t natively support categorical variables, one-hot encoding is performed for categorical variables, with code shown below.

```python
#Creating Dummies for categorical variables
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
```

### Library Used

```python
#library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
```

### Hyperparameter Tuning
This is critical for GBM to prevent over-fitting. Given the size of the dataset, the computational cost to grid search on all parameters is large and time consuming. To overcome this, we have only performed grid search on key parameters, used fix parameters for other parameters and also used only 3-fold cross validation.

Below is a list of parameters used with justification on parameter choice:

- **learning_rate**: It controls how much information from a new tree will be used in the boosting. Value usually ranges between 0 to 1, small values may need more trees to converge however could reduce risks of over-fitting. Thus, we’ve selected a small value of 0.05 and use grid search to find the best ‘n_estimators’ and ‘max_depth’ that correspond to this learning rate.

- **N_estimators**: It controls the maximum number of iterations. We have grid searched on 50, 100 and 200, to find the best parameter.

- **Max_depth**: Controls the maximum depth of the trees. The default value is 6, we have tuned it on small values of 1, 3 and 5 to lower risk of over-fitting.

Code used for hyper parameter tuning is shown below:

```python
n_estimators = [25,50,75,100,200]
max_depth = [1,3,5]

param_grid = {'n_estimators': list(n_estimators),
              'max_depth': list(max_depth)
              }
print("Parameter grid:\n{}".format(param_grid))

#Setup Grid Search
grid_search = GridSearchCV(GradientBoostingClassifier(random_state=1,learning_rate = 0.05)
,verbose = 2, param_grid = param_grid, cv=3, return_train_score=True)
grid_search.fit(X_train, Y_train)
```

Figure below shows the result of grid search. Evident by the chart, ‘max_depth’ of 5 and ‘n_estimators’ of 75 gives the best average score in the testing set of the cross-validation process, thus selected to fit the final model.

![](/GradientBoostingImage.png)

### Final Model
After selecting the parameter, a final model is fitted on all training dataset and results are assessed using the independent testing data.

The top 10 important feature with its importance score produced by the final model is shown in figure below:

![](/FinalModel_GradientBoosting.png)

Code used to fit the final model as below:

```python
# Predicting the Training set results
y_predgb = classifiergb.predict(X_train)
y_predgb_score = classifiergb.predict_proba(X_train)
```

```python
d = {'name':classifiergb.feature_names_in_,'importance_score':classifiergb.feature_importances_}
importance_feature = pd.DataFrame(d).sort_values(by = "importance_score", ascending = False)
importance_feature_top = importance_feature[0:10].sort_values(by = "importance_score", ascending = True)
plt.title("Top 10 important feature \n - Gradient Boost Machine", fontsize=18)
plt.barh(importance_feature_top.name,importance_feature_top.importance_score)
```

### Model Evaluation
Our business problem is to term deposit subscription, as such we will focus on positive class for performance evaluation. Performance results are summarised in the table below, note that Recall, Precision and F1 in the table is based on 50% probability threshold for positive class.


| **Metric**   | **Training** | **Testing** |
|--------------|-------------|------------|
| **Accuracy** | 92.74%     | 91.78%    |
| **Recall**   | 0.583      | 0.519     |
| **Precision**| 0.719      | 0.676     |
| **F1 Score** | 0.644      | 0.587     |
| **AUROC**    | 0.956      | 0.948     |

- There is no significant drop in performance between training and testing data indicating the model is not overfitted. This is also demonstrated in the hyper parameter tuning process where the parameter is chosen to prevent over-fitting.
- We can see the model has high accuracy of 92% this is higher than accuracy can be achieved if predict everything to negative. This means the model has successfully identified a good amount of correct positive and negative classes.
- The precision score for the positive class (Subscribed) is 67%, this means 67% of the predicted positive cases subscribed to the term deposits. The perfect precision score is 100%, 67% is a good performance.
- The recall score for positive class is 52%, it means 52% of clients that will subscribe to a term deposit are picked by the model. This is a good performance, considering only 12% of clients in the dataset subscribed to term deposits. The model works better than randomly selecting a client to target.
- Both recall and precision score are good for class 1 (subscribed), as expected the F1 score which is the harmonic mean of both recall and precision also gives good performance.
- Area Under the Receiver Operating Characteristics (AUROC) is a probability curve that plots the True Positive Rate against the False Positive Rate is also assessed. It is one of the most important metrics to measure the ability of a classifier to distinguish between classes, which ranges from 0.5 to 1. An AUROC of 0.948 indicating the model is excellent at distinguishing between clients that will subscribe or not.

A more detailed classification report and confusion matrix is shown in the figure below:

