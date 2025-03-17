# Bank Marketing Machine Learning Analysis

This report examines a dataset from direct marketing campaigns (phone calls) run by a Portuguese bank.

The aim is to understand what drives successful telemarketing efforts and identify key factors that can improve marketing strategies. The goal is to predict which customers are most likely to sign up for term deposit products through telemarketing. With better targeting, the bank can cut marketing costs and boost return on investment (ROI).

- **Model Type:** Classification model
- **Predictive Target:** Signed up for a term deposit (1) or not (0)

## Data Source

This project uses the **UCI Machine Learning Repository: Bank Marketing Data Set (2012)**. You can access it here:  
[UCI Bank Marketing Dataset](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing) _(Accessed: August 26, 2022)_.

**Original Source:**  
Moro, S., Cortez, P., & Rita, P. (2014). _A Data-Driven Approach to Predict the Success of Bank Telemarketing_. Decision Support Systems, Elsevier, 62, 22-31. doi:10.1016/j.dss.2014.06.001

---

## Training and Testing Data

- **Total Records:** 41,188 (20 attributes)
- **Training Set:** 80% (32,950 records)
- **Testing Set:** 20% (8,238 records)

**Stratified Splitting:** Ensures balanced distribution of the target variable across sets.

---

## Data Preparation and Splitting (Python)

1. Removing Duplicates

- Duplicate records are removed to prevent redundancy and ensure that no data point is overrepresented.

2. Handling Categorical Variables

- The dataset contains categorical features, which must be converted into a numeric format for the model.
- One-hot encoding is applied to categorical variables, ensuring that they are properly represented without introducing bias.

3. Target Variable Transformation

- The dependent variable (y), originally labeled as "yes" or "no", is mapped to binary values (1 and 0, respectively) to enable classification.

4. Handling Missing or Unknown Values

- Some categorical attributes contain "unknown" values, which may represent missing information. These values are replaced with NaN, and the dataset is cleaned by removing rows with missing values.
- Missing values could be imputed rather than removed, depending on the dataset size and the significance of missing data.

5. Feature Scaling

- Since AdaBoost is sensitive to feature magnitudes, StandardScaler is applied to numerical features to standardise them.
- This ensures that all numerical values have similar ranges, improving the stability of the model.

6. Train-Test Split

- The dataset is divided into training (80%) and testing (20%) subsets using stratified sampling, ensuring that the distribution of the target variable is maintained across both sets.

```python
# Import libraries
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
df = pd.read_csv('bank-additional-full.csv', sep=';')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode target ('y'): 'yes' → 1, 'no' → 0
df['y'] = df['y'].map({'yes': 1, 'no': 0}).astype(int)

# Handling missing or unknown values
df.replace("unknown", np.nan, inplace=True)
df.dropna(inplace=True)  # Drop rows with missing values (alternative: impute values)

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

### Justification of model choice

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

| **Metric**    | **Training** | **Testing** |
| ------------- | ------------ | ----------- |
| **Accuracy**  | 92.74%       | 91.78%      |
| **Recall**    | 0.583        | 0.519       |
| **Precision** | 0.719        | 0.676       |
| **F1 Score**  | 0.644        | 0.587       |
| **AUROC**     | 0.956        | 0.948       |

- There is no significant drop in performance between training and testing data indicating the model is not overfitted. This is also demonstrated in the hyper parameter tuning process where the parameter is chosen to prevent over-fitting.
- We can see the model has high accuracy of 92% this is higher than accuracy can be achieved if predict everything to negative. This means the model has successfully identified a good amount of correct positive and negative classes.
- The precision score for the positive class (Subscribed) is 67%, this means 67% of the predicted positive cases subscribed to the term deposits. The perfect precision score is 100%, 67% is a good performance.
- The recall score for positive class is 52%, it means 52% of clients that will subscribe to a term deposit are picked by the model. This is a good performance, considering only 12% of clients in the dataset subscribed to term deposits. The model works better than randomly selecting a client to target.
- Both recall and precision score are good for class 1 (subscribed), as expected the F1 score which is the harmonic mean of both recall and precision also gives good performance.
- Area Under the Receiver Operating Characteristics (AUROC) is a probability curve that plots the True Positive Rate against the False Positive Rate is also assessed. It is one of the most important metrics to measure the ability of a classifier to distinguish between classes, which ranges from 0.5 to 1. An AUROC of 0.948 indicating the model is excellent at distinguishing between clients that will subscribe or not.

A more detailed classification report and confusion matrix is shown in the figure below:

## Model 3: AdaBoostClassifier

AdaBoostClassifier (Adaptive Boosting Classifier) is an ensemble learning algorithm that combines multiple weak classifiers to form a strong predictive model. It works by iteratively training weak learners, adjusting their weights to focus on misclassified instances, and combining their predictions to improve overall accuracy. Unlike traditional models that treat all samples equally, AdaBoostClassifier assigns higher importance to difficult-to-classify examples, allowing it to refine decision boundaries over multiple iterations. By leveraging weak learners such as decision stumps (single-level decision trees), AdaBoostClassifier enhances classification performance while maintaining computational efficiency.

### Justification of Model Choice

- <b>AdaBoostClassifier is effective for classification problems</b>, making it suitable for predicting whether a customer will subscribe to a financial product.
- <b>It enhances weak learners</b> by iteratively refining their predictions, reducing bias and improving overall performance.
- <b>The model can handle both linear and non-linear relationships</b>, allowing it to capture complex patterns in the dataset.
- <b>AdaBoostClassifier performs well with structured tabular data</b>, making it ideal for datasets with a mix of numerical and categorical variables (after one-hot encoding).
- <b>It does not require extensive parameter tuning</b> or strong assumptions about the data, making it a practical and interpretable choice for customer subscription prediction.

### Libraries Used

The implementation of **AdaBoostClassifier** relies on several Python libraries for **data handling, preprocessing, model training, evaluation, and visualisation**. Below is an overview of the libraries used in this study:

- **pandas**: Used for loading, cleaning, and transforming the dataset into a structured format suitable for machine learning.
- **numpy**: Provides support for numerical computations and array operations, ensuring efficient data processing.
- **seaborn**: Used for advanced data visualisation, particularly for analysing feature distributions and plotting confusion matrices.
- **matplotlib.pyplot**: Used for generating various plots, including feature importance graphs, ROC curves, and confusion matrices.
- **sklearn.model_selection**: Provides **train_test_split** for splitting the dataset into training and testing sets, and **RandomizedSearchCV** for hyperparameter tuning.
- **sklearn.preprocessing**: Includes **StandardScaler**, which is used to standardise numerical features, ensuring uniform feature scaling.
- **sklearn.ensemble**: Contains **AdaBoostClassifier**, the core model used for classification in this study.
- **sklearn.metrics**: Provides multiple evaluation metrics, including **accuracy_score, precision_score, recall_score, f1_score, roc_auc_score**, as well as functions for generating **classification reports, confusion matrices, and ROC curves**.

These libraries collectively enable the efficient development, tuning, and evaluation of the **AdaBoostClassifier** while ensuring that the model is both interpretable and well-optimised.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
```

### Data Pre-Processing

The dataset used in this study has already undergone extensive preprocessing, as outlined in the group section. This includes <b> removing duplicate records, handling missing or unknown values, encoding categorical variables through one-hot encoding </b>, and <b> splitting the dataset into training and testing sets</b> using stratified sampling to maintain class distribution. These steps ensure that the data is well-structured and suitable for classification.

For this specific implementation of <b>AdaBoostClassifier</b>, an additional preprocessing step is applied — <b>Feature Scaling</b>. Since AdaBoost is sensitive to feature magnitudes, standardisation is performed using <b>StandardScaler</b>, transforming numerical features to have <b>zero mean and unit variance</b>. This prevents attributes with larger scales from disproportionately influencing the learning process. The transformation is applied as follows:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Applying standardisation ensures that numerical attributes such as <b>age, balance, and duration</b> remain on a consistent scale, contributing to improved model stability and performance. With these preprocessing steps in place, the dataset is fully prepared for training the <b>AdaBoostClassifier</b>, enabling an effective and unbiased evaluation of its predictive capabilities.

### Baseline Model Training and Evaluation

To establish a benchmark for comparison, an initial AdaBoostClassifier model is trained using default hyperparameters, serving as a baseline for further refinement. The dataset is divided into training and test sets using stratified sampling to maintain the target variable's distribution. The model is trained with 50 estimators (default setting) and a learning rate of 1.0, ensuring that the model can effectively learn from the data while maintaining computational efficiency.

The classification report for the baseline model highlights that while precision for the majority class is high, recall for the minority class (subscribed customers) is comparatively lower, suggesting that the model struggles to capture all positive cases. The confusion matrix further illustrates this issue, as it shows a substantial number of false negatives. Despite this, the ROC-AUC score of 0.9358 suggests that the model has strong overall predictive capability, meaning it can distinguish between subscribed and non-subscribed customers well.

#### Step 1: Define Model Evaluation Function

Before training, a function is created to evaluate models by computing classification metrics, plotting a confusion matrix, and generating a ROC curve:

```python
# Model Evaluation Function
def evaluate_model(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nPerformance for {label}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {label}")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label}")
    plt.legend()
    plt.show()
```

#### Step 2: Train the Baseline Model

The baseline model was implemented as follows:

```python
# Train baseline model
baseline_model = AdaBoostClassifier(n_estimators = 50, random_state=42)
baseline_model.fit(X_train, y_train)
```

#### Step 3: Evaluate the Baseline Model

After training, the baseline model was evaluated using key classification metrics, a confusion matrix, and a ROC curve:

```python
# Evaluate baseline model
evaluate_model(baseline_model, X_test, y_test, "Baseline Model")
```

<b>Baseline Model Performance</b>

The following evaluation metrics summarise the model’s performance:

![](images/adaboostclassifier/baseline_model_performance.png)

<b> Confusion Matrix - Baseline Model</b>

The confusion matrix visualises the number of correct and incorrect predictions:

- <b>True Negatives (TN): 5178</b> (Correctly predicted non-subscribed customers)
- <b>False Positives (FP): 146</b> (Incorrectly predicted non-subscribers as subscribers)
- <b>False Negatives (FN): 468</b> (Incorrectly predicted subscribers as non-subscribers)
- <b>True Positives (TP): 304</b> (Correctly predicted subscribed customers)

![](images/adaboostclassifier/confusion_matrix_baseline.png)

<b>ROC Curve - Baseline Model</b>

The ROC curve demonstrates the model’s ability to distinguish between subscribed and non-subscribed customers. The <b>AUC score of 0.9358</b> indicates strong classification performance.

![](images/adaboostclassifier/roc_curve_baseline.png)

### Baseline Model Observation

- The <b>high precision (92%) for the majority class</b> suggests that the model correctly predicts non-subscribed customers with high accuracy.
- However, the <b>recall for subscribed customers is low (39%)</b>, indicating that the model fails to identify a significant proportion of actual subscribers.
- The <b>high ROC-AUC score (0.9358)</b> suggests that despite the recall issue, the model effectively distinguishes between the two classes.

This baseline model serves as a <b>reference point</b> for further improvements through <b>hyperparameter tuning</b>, which aims to increase recall and F1-score while maintaining strong overall classification performance.

### Hyperparameter Tuning

To improve the predictive performance of the AdaBoostClassifier, hyperparameter tuning is conducted using RandomizedSearchCV. The goal is to optimise the number of estimators and the learning rate, enhancing the model’s ability to identify subscribed customers while minimising false negatives.

#### Step 4: Perform Hyperparameter Tuning

To efficiently search for the best hyperparameters, a range of values is defined for n_estimators and learning_rate. The tuning process explores the following values:

- `n_estimators = [50, 100, 200, 300, 400, 500]`
- `learning_rate = [0.001, 0.01, 0.1, 0.5, 1.0]`

Using RandomizedSearchCV, the algorithm selects a subset of hyperparameter combinations to reduce computation time while ensuring model optimisation.

```python
# Hyperparameter Tuning with RandomizedSearchCV for efficiency
param_dist = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
}

random_search = RandomizedSearchCV(AdaBoostClassifier(random_state=42), param_distributions=param_dist, cv=5, scoring='f1', n_jobs=-1, n_iter=10, random_state=42)
random_search.fit(X_train, y_train)
print(f"Best Parameters: {random_search.best_params_}")
```

After running the tuning process, the <b>best combination of hyperparameters is selected</b> and stored in `random_search.best_params_`, which will be used to train the final model.

### Hyperparameter Tuning

After optimising the AdaBoostClassifier, the final model is trained using the best hyperparameters identified through RandomizedSearchCV. This refined model is expected to improve recall and F1-score, leading to better identification of subscribed customers.

#### Step 5: Train the Final Model

The best parameters found from the tuning process are now used to train the final AdaBoostClassifier, ensuring an optimal balance between precision and recall.

```python
# Fit final model with best parameters
final_model = random_search.best_estimator_
final_model.fit(X_train, y_train)
```

By retraining the model with optimised parameters, we expect to see improvements in its ability to classify positive cases more effectively.

#### Step 6: Evaluate the Final Model

After training the final model, it is evaluated using the same function applied to the baseline model. The classification report, confusion matrix, and ROC curve are generated to assess improvements in performance.

```python
# Evaluate final models
evaluate_model(final_model, X_test, y_test, "Final Model (After Tuning)")
```

<b>Final Model Observation</b>

The following evaluation metrics summarise the model’s performance after hyperparameter tuning:

![](images/adaboostclassifier/final_model_performance.png)

<b>Confusion Matrix - Final Model</b>

The confusion matrix visualises the number of correct and incorrect predictions:

- <b>True Negatives (TN): 5159</b> (Correctly predicted non-subscribed customers)
- <b>False Positives (FP): 165</b> (Incorrectly predicted non-subscribers as subscribers)
- <b>False Negatives (FN): 457</b> (Incorrectly predicted subscribers as non-subscribers)
- <b>True Positives (TP): 315</b> (Correctly predicted subscribed customers)

![](images/adaboostclassifier/confusion_matrix_final.png)

<b> ROC Curve - Final Model</b>

The ROC curve demonstrates the model’s ability to distinguish between subscribed and non-subscribed customers. The AUC score of 0.9378 indicates strong classification performance, slightly improving over the baseline model.

![](images/adaboostclassifier/roc_curve_final.png)

### Final Model Observations

- <b>Recall has improved slightly</b>, increasing from <b>39.38% (baseline model)</b> to <b>40.80% (final model)</b>. This means the model identifies a <b>higher number of actual subscribers.</b>
- <b>F1-score has increased</b>, indicating that the final model <b>better balances precision and recall.</b>
- <b>The ROC-AUC score increased slightly</b>, reinforcing the model’s effectiveness at distinguishing between subscribed and non-subscribed customers.
- <b>Precision has slightly decreased</b>, suggesting that the model is slightly more prone to false positives in order to improve recall.

Overall, <b>hyperparameter tuning has successfully optimised the AdaBoostClassifier</b>, improving its ability to correctly identify customers who are likely to subscribe, while maintaining strong classification accuracy.

### Feature Importance Analysis

To understand which factors contribute the most to customer subscription decisions, a feature importance analysis is conducted. The AdaBoostClassifier assigns different weights to features based on their impact on improving classification accuracy. This analysis provides valuable insights into which attributes play a key role in predicting customer subscriptions, allowing for targeted marketing and customer engagement strategies.

The following code extracts and plots the feature importances assigned by the final AdaBoostClassifier:

```python
# Feature Importance Analysis
feature_importances = pd.Series(final_model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind="bar", figsize=(12,6), title="Feature Importance in Final Model")
plt.show()
```

The feature importance plot highlights the most influential factors in predicting customer subscription outcomes.

![](images/adaboostclassifier/feature_importance.png)

1. <b>Duration of the Call</b> is the most critical predictor, indicating that <b>the length of the customer’s last phone call significantly impacts the likelihood of subscription.</b>
2. <b>Economic indicators</b>, such as <b>euribor3m (Euro interbank interest rate), employment variation rate, and consumer confidence index</b>, are also influential, showing that <b>macroeconomic conditions affect customer decisions.</b>
3. <b>Customer demographics</b>, such as <b>age and previous campaign interactions</b>, play a notable role in subscription likelihood.
4. <b>One-hot encoded categorical variables</b> such as <b>job type, education, and marital status</b> contribute less but still have an impact.

### Model Evaluation: Baseline Model vs Final Model

To assess the impact of hyperparameter tuning, the baseline AdaBoostClassifier model is compared to the final optimised model. This evaluation focuses on key classification metrics, including accuracy, precision, recall, F1-score, and ROC-AUC, to determine whether the optimised model has improved over the initial version.

The table below summarises the key performance metrics for both models:

| **Metric**    | **Baseline Model** | **Final Model** |
| ------------- | ------------------ | --------------- |
| **Accuracy**  | 89.93%             | 89.80%          |
| **Recall**    | 67.76%             | 65.62%          |
| **Precision** | 39.38%             | 40.80%          |
| **F1 Score**  | 49.75%             | 50.32%          |
| **AUROC**     | 93.58%             | 93.78%          |

These results indicate that hyperparameter tuning led to a slight increase in recall and F1-score, making the final model better at capturing actual subscribers. However, this comes at the cost of a slight decrease in precision, meaning the model produces more false positives.

- Recall improved from 39.38% to 40.80%, meaning the final model correctly identifies more subscribed customers (True Positives) compared to the baseline model.
- F1-score increased from 49.75% to 50.32%, indicating a better balance between precision and recall, leading to a more effective classification model.
- Precision slightly decreased from 67.56% to 65.62%, which suggests a small increase in false positives as the model prioritises recall over precision.
- The ROC-AUC score increased from 93.58% to 93.78%, reinforcing the model’s ability to differentiate between customers who will and will not subscribe.
- The confusion matrix shows a reduction in false negatives, suggesting that the final model has improved its ability to capture actual subscribers while maintaining a high number of correctly predicted non-subscribers.
- The final model demonstrates a slight trade-off, sacrificing some precision to achieve higher recall, which is beneficial in cases where identifying potential subscribers is more important than minimising false positives.
- For the ROC Curve Model, the increase in AUC from 0.9358 to 0.9378 suggests a slight improvement in the model’s ability to distinguish between subscribed and non-subscribed customers.

Hyperparameter tuning has successfully improved the model’s ability to detect actual subscribers by increasing recall and F1-score while maintaining a high ROC-AUC score. While precision has slightly decreased, the trade-off allows the model to correctly identify more customers who are likely to subscribe. This suggests that the final model is a more effective predictive tool for customer subscription classification, especially in scenarios where capturing all potential subscribers is a higher priority than minimising false positives.

### Model Evaluation: Final Model (Train Set) vs Final Model (Test Set)

After training the final AdaBoostClassifier model, it is essential to compare its performance on the training set and the test set to assess how well it generalises to unseen data. A significant discrepancy between these two evaluations may indicate overfitting, where the model performs well on training data but struggles with new data.

The table below summarises the key performance metrics for both models:

| **Metric**    | **Final Model (Train)** | **Final Model (Test)** |
| ------------- | ----------------------- | ---------------------- |
| **Accuracy**  | 90.59%                  | 89.80%                 |
| **Recall**    | 69.74%                  | 65.62%                 |
| **Precision** | 45.33%                  | 40.80%                 |
| **F1 Score**  | 54.95%                  | 50.32%                 |
| **AUROC**     | 94.44%                  | 93.78%                 |

While there is a minor drop in performance from the train set to the test set, the ROC-AUC score remains high, suggesting that the model still maintains strong classification ability when generalising to unseen data.

- Performance drops slightly from training to test data, which is expected but does not indicate severe overfitting.
- Accuracy decreases from 90.59% (train) to 89.80% (test), showing a small generalisation gap.
- Recall drops from 45.33% (train) to 40.80% (test), meaning the model is slightly less confident in predicting actual subscribers in unseen data.
- Precision is slightly lower on test data (65.62%) compared to training (69.74%), meaning there are slightly more false positives when generalising to new samples.
- The ROC-AUC score remains high (0.9444 for train, 0.9378 for test), confirming strong classification performance across both sets.

The final model generalises well to unseen data, as the test performance remains close to the training performance. The slight decrease in precision and recall suggests that while the model performs well, there is still room for improvement in identifying actual subscribers. However, the high ROC-AUC scores (0.9444 for train, 0.9378 for test) indicate that the model maintains strong overall classification ability. The results suggest that the final AdaBoostClassifier model is effective in predicting customer subscriptions without significant overfitting.

## Model 4: Support Vector Machine

## Model Comparison

The table below summarises the performance of the different classification models evaluated. While Random Forest and Gradient Boosting (GBM) exhibit the highest overall performance across all five metrics, AdaBoost achieves competitive precision but falls behind in recall and F1-score. The Support Vector Machine (SVM) model, despite having strong precision, struggles with recall and AUROC, indicating limited ability to distinguish between subscribed and non-subscribed customers.

| **Model**                  | **Accuracy** | **Recall** | **Precision** | **F1 Score** | **AUROC** |
| -------------------------- | ------------ | ---------- | ------------- | ------------ | --------- |
| **AdaBoost Classifier**    | 89.80%       | 40.80%     | 65.62%        | 50.32%       | 93.78%    |
| **GBM**                    | 92%          | 52%        | 68%           | 59%          | 94.9%     |
| **Random Forest**          | 92%          | 51%        | 67%           | 58%          | 94.2%     |
| **Support Vector Machine** | 91%          | 38%        | 65%           | 49%          | 67.6%     |

## Model Strengths, Weaknesses, and Improvements

The table below summarises the key strengths, weaknesses, and potential improvements for each classification model. While Random Forest and GBM perform well in various aspects, AdaBoost (ABC) offers competitive results but requires optimisation to handle imbalanced data effectively.

| **Model**                        | **Strengths**                                                                                                                                                                                                                                                                                                                                                       | **Weaknesses**                                                                                                                                                                                                                                                                                 | **Potential Improvements**                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AdaBoost Classifier**          | ✅ **Automatically prioritises important features**, helping in feature selection. <br> ✅ **Boosting approach refines weak learners**, leading to **strong generalisation**. <br> ✅ **Maintains high AUROC score (93.78%)**, showing strong classification ability. <br> ✅ **Does not require variable scaling or transformation**, making preprocessing easier. | ❌ **Low recall (40.80%)**, meaning it fails to identify a large proportion of actual subscribers. <br> ❌ **Prone to overfitting**, especially if weak learners are too complex. <br> ❌ **Less interpretable**, as multiple iterations of weak classifiers make it harder to explain.        | 🔹 **Class imbalance solutions:** Apply **SMOTE (Synthetic Minority Over-sampling Technique)** to improve recall. <br> 🔹 **Improve weak learners:** Use **pruned decision trees** instead of simple stumps for better performance. <br> 🔹 **Advanced hyperparameter tuning:** Experiment with **learning rate adjustments** and **alternative base classifiers** (e.g., logistic regression). |
| **Gradient Boosting (GBM)**      | ✅ **Performs feature selection automatically**, reducing irrelevant variables. <br> ✅ **High predictive accuracy** and **strong performance in non-linear relationships**. <br> ✅ **Provides probability estimates**, making it more useful for business decisions. <br> ✅ **Works well with imbalanced datasets** compared to simpler models.                  | ❌ **Difficult to interpret**, making it challenging to explain model decisions. <br> ❌ **Prone to overfitting**, especially with noisy data or poorly tuned parameters. <br> ❌ **Computationally expensive**, requiring extensive training time.                                            | 🔹 **Hyperparameter tuning:** Conduct an extensive **grid search for optimal learning rates, boosting iterations, and tree depths**. <br> 🔹 **Model interpretability:** Use **SHAP (Shapley Additive Explanations)** or **partial dependence plots** to improve explainability.                                                                                                                |
| **Random Forest**                | ✅ **Resilient to overfitting**, making it a stable model for generalisation. <br> ✅ **Performs automatic feature selection**, reducing the need for manual preprocessing. <br> ✅ Can handle **both linear and non-linear relationships** effectively. <br> ✅ **Works well with missing data** and does not require variable scaling.                            | ❌ **Difficult to interpret**, as decision trees form a complex ensemble. <br> ❌ **Hyperparameter tuning is challenging**, requiring careful selection of tree depth, number of trees, and sample splits. <br> ❌ **Computationally expensive**, particularly with large datasets.            | 🔹 **Feature engineering:** Incorporate additional **macroeconomic indicators** to improve prediction accuracy. <br> 🔹 **Optimised hyperparameter tuning:** Use **GridSearchCV or RandomizedSearchCV** to fine-tune the number of estimators and tree depth.                                                                                                                                   |
| **Support Vector Machine (SVM)** | ✅ **Effective in high-dimensional spaces**, making it suitable for complex datasets. <br> ✅ **Flexible kernel functions** allow modelling of **both linear and non-linear relationships**. <br> ✅ **Robust against overfitting** with appropriate regularisation.                                                                                                | ❌ **Computationally expensive**, especially for large datasets, leading to long training times. <br> ❌ **Does not provide probability estimates directly**, making probability-based decisions harder. <br> ❌ **Hard to interpret**, particularly with **high-dimensional feature spaces**. | 🔹 **Dimensionality reduction:** Apply **Principal Component Analysis (PCA)** to reduce feature space for faster training. <br> 🔹 **Handle class imbalance:** Assign **class weights** to improve recall on minority classes.                                                                                                                                                                  |

## Conclusion

The results of our analysis indicate that Gradient Boosting Machine (GBM) and Random Forest performed the best, achieving the highest accuracy, recall, and AUROC scores. AdaBoost, while competitive in precision and AUROC, struggled with recall and overall predictive performance, making it less effective for identifying actual subscribers. Support Vector Machine (SVM) had the weakest overall performance, particularly in AUROC, indicating poor discrimination between subscribed and non-subscribed customers.

Among all models, GBM is recommended for deployment due to its ability to balance high predictive accuracy with feature selection capabilities. Additionally, GBM provides probability estimates, making it more actionable for business strategies compared to models like SVM, which produce only discrete predictions (i.e., 1 or 0). This probability-based approach allows for risk-based decision-making, such as prioritising high-probability customers for targeted marketing efforts.

### Key Findings

- Gradient Boosting and Random Forest outperformed all other models in terms of accuracy, recall, and AUROC, making them the strongest candidates for customer subscription prediction.
- AdaBoost had relatively low recall (40.80%) and F1-score (50.32%), meaning it failed to correctly identify a large proportion of actual subscribers. This suggests that AdaBoost struggled with imbalanced data, missing potential subscribers while maintaining high AUROC (93.78%).
- SVM had the weakest AUROC (67.6%), indicating that it was the least effective at distinguishing between subscribed and non-subscribed customers.
- The most influential features in predicting customer subscriptions were:
  - Duration: The length of the last contact with the customer (in seconds), which remains the most dominant predictor.
  - Number of employees (nr.employed): A macroeconomic indicator that reflects overall financial conditions.
  - Euribor 3-month rate (euribor3m): A key interest rate influencing banking and financial decisions.
  - Previous campaign interaction (pdays): The number of days since the customer’s last interaction with a previous marketing campaign.

These findings highlight the importance of customer engagement, economic conditions, and prior interactions in determining a customer’s likelihood to subscribe.

### Future Work and Recommendations

To further refine and improve the predictive model, the following steps should be taken:

1. Addressing Class Imbalance and Improving Recall

- Since AdaBoost and SVM performed poorly in recall, future work should explore techniques such as:
  - Synthetic Minority Over-sampling Technique (SMOTE) to rebalance the dataset.
  - Class-weighted loss functions to reduce bias toward the majority class.
- Improving recall is critical in subscription prediction, as businesses want to identify all potential customers, even at the cost of some false positives.

2. Regular Model Updating and Calibration

- Periodic retraining should be implemented to adapt to changes in customer behaviour and economic conditions.
- A monitoring framework should be established to detect performance drift, ensuring the model remains accurate over time.

3. Expanding the Dataset for Better Predictive Power

- The current dataset lacks financial history and customer-specific economic variables. Future datasets should include:
  - Income level to determine affordability.
  - Existing loan or debt status to assess financial obligations.
  - Spending behaviour and transaction history to evaluate customer engagement with financial services.

4. Incorporating Time-Series Analysis

- The dataset only covers slightly over a year, making it difficult to model long-term economic effects on subscription behaviour.
- Expanding the dataset to multiple years would allow for:
  - Time-series forecasting to predict customer behaviour in changing economic conditions.
  - Dynamic adaptation to macroeconomic shifts, improving model robustness.
