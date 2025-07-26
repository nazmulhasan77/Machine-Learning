The **K-Nearest Neighbors (KNN)** algorithm is a **supervised machine learning** algorithm used for **classification** and **regression**, but it's more commonly used for classification problems.

---
![image.png](attachment:image.png)(https://miro.medium.com/v2/resize:fit:600/0*6PwJ2ipbI_S9bMHr.png)

### 🔷 How KNN Works:

1. **Training Phase:**

   * KNN is a **lazy learner** – it doesn't explicitly learn a model.
   * It just **stores the training data**.

2. **Prediction Phase (for a new data point):**

   * Compute the **distance** (usually Euclidean) between the new point and all points in the training data.
   * **Select the K nearest neighbors** (K is a user-defined number).
   * **Vote**:

     * **Classification**: Assign the most common class among the K neighbors.
     * **Regression**: Take the average (or weighted average) of the K neighbors’ values.

---


### 🔷 Euclidean Distance Formula:

For two points:

$$
P = (p_1, p_2, ..., p_n), \quad Q = (q_1, q_2, ..., q_n)
$$

The **Euclidean distance** between $P$ and $Q$ is:

$$
d(P, Q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \cdots + (p_n - q_n)^2}
$$

---

### 🔹 2D Example:

Let:

$$
P = (3, 4), \quad Q = (7, 1)
$$

Then:

$$
d(P, Q) = \sqrt{(3 - 7)^2 + (4 - 1)^2} = \sqrt{(-4)^2 + (3)^2} = \sqrt{16 + 9} = \sqrt{25} = 5
$$

### 🔶 Key Concepts:

| Concept             | Details                                                                            |
| ------------------- | ---------------------------------------------------------------------------------- |
| **K**               | Number of nearest neighbors to consider. Choosing the right K is critical.         |
| **Distance Metric** | Common: Euclidean, Manhattan, Minkowski, Hamming (for categorical data).           |
| **Feature Scaling** | Important! Features should be normalized/standardized due to distance calculation. |

---

### 🔹 Example (Classification):

Suppose we want to classify a fruit as **apple** or **orange** based on features like weight and color.

If K=3 and among the 3 nearest neighbors, 2 are apples and 1 is orange → classify as **apple**.

---



### ✅ Advantages:

* Simple to implement and understand.
* No assumptions about data distribution.
* No training phase needed.
* Can be effective with enough representative data.

### ❌ Disadvantages:

* Computationally expensive for large datasets (since distance to every point must be computed).
* Sensitive to irrelevant or redundant features.
* Poor performance on imbalanced data or high-dimensional data (curse of dimensionality).