# 🌳 Decision Trees in Machine Learning

### ✅ What is a Decision Tree?

A **decision tree** is a flowchart-like structure used for classification and regression. It splits the dataset into subsets based on feature values, aiming to make the target variable as pure as possible in each split.

---

## 📚 Theory

### 1. **Entropy (Measure of Impurity)**

Entropy tells us how "mixed" the classes are:

$$
\text{Entropy}(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)
$$

* $p_i$: Proportion of class $i$ in the set
* Range: 0 (pure) to 1 (maximum impurity for binary)

#### Example:

If a set has 50% class 0 and 50% class 1, entropy = 1 (most impure).
If a set has 100% of one class, entropy = 0 (pure).

---

### 2. **Information Gain (IG)**

Information Gain is the reduction in entropy after splitting: 

$$
IG(S, A) = Entropy(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \cdot Entropy(S_v)
$$

* $S$: Current dataset
* $A$: Attribute to split on
* $S_v$: Subset where attribute $A = v$

The attribute with **highest IG** is chosen to split.

---

### 3. **ID3 Algorithm (Iterative Dichotomiser 3)**

**Steps:**

1. Calculate entropy of current dataset.
2. For each attribute, calculate **information gain**.
3. Choose the attribute with **highest gain**.
4. Split data, repeat for each subset until:

   * All data in subset belongs to one class
   * No more attributes to split

---

## 🛠️ Python Implementation (Using Sklearn)

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train a decision tree
model = DecisionTreeClassifier(criterion='entropy')  # ID3 uses entropy
model.fit(X, y)

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree Using ID3 (Entropy)")
plt.show()
```

---

## 📌 Summary

| Concept           | Description                              |
| ----------------- | ---------------------------------------- |
| **Entropy**       | Measures disorder/impurity in dataset    |
| **Info Gain**     | How much entropy is reduced by a feature |
| **ID3 Algorithm** | Builds tree using max info gain splits   |

---