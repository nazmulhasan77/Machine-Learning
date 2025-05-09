# 🤖 Logistic Regression in Machine Learning

## 🔍 Overview

**Logistic Regression** is a **supervised learning algorithm** used for **binary classification** tasks like **yes/no, true/false, 0/1.**

---

## 📐 Key Concepts

- **Purpose**: Predict whether an input belongs to class 1 or class 0.
- **Output**: A probability value between **0 and 1**.
- **Threshold**: Usually 0.5 – if the output ≥ 0.5 → class 1, else class 0.

### ✅ Logistic Regression Equation

$$
P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n)}}
$$


This is the **sigmoid function** applied to a **linear combination** of input features.

---

## ⚙️ How It Works

1. Takes input features (e.g., age, income).
2. Computes a weighted sum: \( z = \beta_0 + \beta_1X_1 + \dots + \beta_nX_n \)
3. Applies the sigmoid function: \( \sigma(z) = \frac{1}{1 + e^{-z}} \)
4. Converts probability to class based on threshold (e.g. 0.5)

---

## 🔄 Machine Learning Pipeline with Logistic Regression

### 1. Collect Data
- Example: Dataset with features like age, income, and purchase (0 = no, 1 = yes)

### 2. Preprocess the Data
- Handle missing values
- Normalize or scale features
- Encode categorical variables

### 3. Split the Dataset
- **Training set** (e.g., 80%)
- **Test set** (e.g., 20%)

### 4. Train the Model
- Learn weights using **gradient descent**
- Minimize the **log loss function**

### 5. Make Predictions
- Output probabilities → classify based on threshold

### 6. Evaluate Performance
Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

## 📈 Decision Boundary

Logistic Regression finds a **linear decision boundary** in feature space:
- For 2 features → straight line
- For more → hyperplane

---

## 💻 Python Code Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Step 1: Load dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Step 2: Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Step 6: Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
```

## 📌 Use Cases

- **Email spam detection**  
  Classify emails as spam or not spam.

- **Credit risk modeling**  
  Predict whether a person is likely to default on a loan.

- **Disease diagnosis**  
  Example: Classify tumors as malignant or benign.

- **Customer churn prediction**  
  Predict if a customer is likely to leave a service.

---

## ⚠️ Limitations

- Assumes a **linear relationship** between input features and the **log-odds** of the output.
- Not suitable for **non-linear** or **complex data** like images or speech.
- **Sensitive to outliers** and **multicollinearity** among features.
