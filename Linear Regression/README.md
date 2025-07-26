# 📊 Linear Regression Overview

## 🧠 What is Linear Regression ?

Linear Regression is a **supervised machine learning** algorithm used to **predict a continuous value** based on the relationship between one or more features (input variables) and a label (output variable). It assumes a **linear relationship** between them.

---

## 🧮 Formula

Simple Linear Regression is expressed as:

<b>y = mx + c </b>


Where:
- `y` = predicted value  
- `x` = input feature  
- `m` = slope (coefficient)  
- `c` = y-intercept  
---

## 📘 Types of Linear Regression

### ✅ Simple Linear Regression
- Uses **one** input variable  
- Example: `house size` → `price`

### ✅ Multiple Linear Regression
- Uses **multiple** input variables  
- Example: `size + rooms + location` → `price`

---

## 🎯 Goal

The goal is to find the **best-fitting straight line** through the data by minimizing the prediction error using **Least Squares**.

---

## 📌 Use Cases

- Predicting house prices
- Sales forecasting
- Stock market trends
- Student performance prediction

---

## 📷 Visualization (Optional)

<img src="Experiance & Salary Actual vs Best Fit Line.png" width="400px" alt="ML Types">

---

## 🛠️ Example (Python)

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
prediction = model.predict([[6]])
print("Prediction for x=6:", prediction)


