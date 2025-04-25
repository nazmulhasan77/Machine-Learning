# Machine Learning (ML)

Machine Learning (ML) is a branch of artificial intelligence that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed.

> "Father of Machine Learning" is often attributed to Arthur Samuel.

## Types of Machine Learning

### 1. Supervised Learning
- Trained using <b>labeled data </b> (input-output pairs).
- Learns a mapping from inputs (features) to outputs (labels).
- Common use-cases: Classification & Regression.

#### Examples:
- Linear Regression
- Decision Trees
- Support Vector Machines (SVM)
- Neural Networks

#### 1.1 🔹 Classification Problem
- Goal: Predict a <b>category/class label.</b>
- **Examples:**
  - Email: spam or not spam
  - Image: Identify cat, dog, or bird
  - Medical: Disease present or not
- **Output:** Discrete values
- **Algorithms:**
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Support Vector Machines
  - Neural Networks

#### 1.2🔹 Regression Problem
- Goal: Predict a <b>continuous value. </b>
- **Examples:**
  - House prices based on features
  - Estimating age from photo
  - Forecasting stock prices
- **Output:** Continuous values
- **Algorithms:**
  - Linear Regression
  - Ridge/Lasso Regression
  - Decision Trees
  - SVR (Support Vector Regression)
  - Neural Networks

##### Notable Techniques:
- **1.1 Bayes' Theorem** for spam detection with "Free"
- **1.2 Linear Regression** (Least Squares Method)

### 2. Unsupervised Learning
- Trained using unlabeled data.
- Finds hidden structures or patterns in data.
- Common use-cases: Clustering & Association Rules.

#### Examples:
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- Autoencoders

#### 2.1🔹 Clustering Problem
- Goal: Group similar items together without labels.
- **Examples:**
  - Customer Segmentation
  - Social Network Analysis
  - Document/News Clustering
  - Image Segmentation
  - Anomaly Detection

#### 🔹2.2 PCA (Principal Component Analysis) Problem
- Goal: Reduce dimensionality while preserving variance.
- **Examples:**
  - Data Visualization
  - Noise Reduction
  - Speeding up ML Models
  - Gene Expression Analysis

### 3. Reinforcement Learning (RL)
- Learns by interacting with an environment and receiving feedback.
- Goal: Maximize cumulative reward.
- Common in robotics, games, self-driving cars.

#### Key Concepts:
- Agent
- Environment
- Actions
- Rewards

#### Examples:
- Q-Learning
- Deep Q Networks (DQN)
- Policy Gradient Methods (e.g., A3C, PPO)

---

## Training, Validation, and Test Data

| Type        | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| Training    | Model learns from this data. Fits model parameters.                        |
| Validation  | Used to tune model & hyperparameters. Prevents overfitting.                |
| Test        | Evaluates final model performance. Simulates real-world generalization.    |

### 🔹 80/20 Train-Test Split
1. Split data:
   - 80% Training Set
   - 20% Test Set
2. Process:
   - Prepare dataset
   - Random split
   - Train model
   - Evaluate on test data

---

## Performance Metrics

| Problem Type   | Common Metrics                            |
|----------------|--------------------------------------------|
| Classification | Accuracy, F1 Score, ROC-AUC, Precision     |
| Regression     | MAE, MSE, RMSE, R² Score                   |
| Clustering     | Silhouette Score, Inertia                  |

---

## ✅ Machine Learning Algorithms Table

| Regression           | Classification               | Clustering           |
|----------------------|-------------------------------|----------------------|
| Linear Regression    | Logistic Regression           | K-Means Clustering   |
|                      | Support Vector Machine (SVM) |                      |
|                      | K-Nearest Neighbors (KNN)     |                      |
|                      | Naive Bayes                   |                      |
|                      | Neural Networks               |                      |

