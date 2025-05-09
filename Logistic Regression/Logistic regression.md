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
- <b>Encode</b> categorical variables

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
# <b> Encoding </b>

Logistic regression, like many machine learning algorithms, requires **numerical input**. When your dataset contains **categorical features**, you need to **encode** them before using logistic regression. Here's a breakdown of common encoding techniques and when to use them:

---

### 🔹 1. **One-Hot Encoding (OHE)**

**Best for:** Nominal categorical variables (no natural order)
**How it works:** Creates a new binary column for each category.

| Color | One-Hot Encoded |
| ----- | --------------- |
| Red   | \[1, 0, 0]      |
| Green | \[0, 1, 0]      |
| Blue  | \[0, 0, 1]      |

**Tools:**

* `pandas.get_dummies()`
* `sklearn.preprocessing.OneHotEncoder`

---

### 🔹 2. **Label Encoding**

**Best for:** Ordinal categorical variables (ordered categories)
**How it works:** Assigns a unique integer to each category.

| Size   | Encoded |
| ------ | ------- |
| Small  | 0       |
| Medium | 1       |
| Large  | 2       |

**Tools:**

* `sklearn.preprocessing.LabelEncoder`
  ⚠️ Use with caution: Logistic regression may interpret this as numeric *order* or *distance*, which may not be appropriate for nominal variables.

---

### 🔹 3. **Binary Encoding / Hash Encoding**

**Best for:** High-cardinality categorical variables

* Binary encoding reduces dimensionality by converting category indices to binary code.
* Hashing (e.g., `FeatureHasher`) hashes category strings to a fixed-size vector.

**Tools:**

* `category_encoders` library (`BinaryEncoder`, `HashingEncoder`)

---

### 🚫 Avoid Direct Integer Encoding for Nominal Variables

Assigning integers (e.g., Red=1, Blue=2) without considering their nature can mislead logistic regression, as it may infer a false ordering.

---

### 🛠️ Example with `pandas` and `sklearn`

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Sample data
df = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Blue'],
    'Label': [1, 0, 1, 0]
})

# One-hot encode
X = pd.get_dummies(df['Color'], drop_first=True)
y = df['Label']

# Train logistic regression
model = LogisticRegression()
model.fit(X, y)
```

Would you like a real-world dataset example using logistic regression with encoded features?


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
