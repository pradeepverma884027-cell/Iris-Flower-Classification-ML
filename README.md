# 🌸 Iris Flower Classification using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
---

## 📖 Project Overview

This project is a Machine Learning application that classifies Iris flowers into three different species using the **K-Nearest Neighbors (KNN)** algorithm.

The application is built using **Python**, **Scikit-Learn**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **Streamlit**.

The model predicts the flower species based on four flower measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## 🎯 Objectives

- Learn Machine Learning Classification
- Understand K-Nearest Neighbors Algorithm
- Perform Data Preprocessing
- Visualize the Dataset
- Train and Evaluate the Model
- Deploy the Model using Streamlit

---

# 🌺 Iris Flower Classes

- Setosa
- Versicolor
- Virginica

---

# 📂 Project Structure

```text
Iris-Flower-Classification-ML/

│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── dataset/
│     iris.csv
│
├── model/
│     knn_model.pkl
│     scaler.pkl
│
├── src/
│     preprocessing.py
│     visualization.py
│     evaluation.py
│
├── outputs/
│     confusion_matrix.png
│     pairplot.png
│     heatmap.png
│     accuracy_vs_k.png
│
├── screenshots/
│
└── notebooks/
      Iris_Classification.ipynb
```

---

# 📊 Dataset

Dataset Used:

**Iris Flower Dataset**

- Total Samples : 150
- Features : 4
- Classes : 3

Features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target:

- Setosa
- Versicolor
- Virginica

---

# ⚙ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit
- Joblib

---

# 🚀 Machine Learning Workflow

```
Load Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Scaling
        │
        ▼
Train-Test Split
        │
        ▼
Train KNN Model
        │
        ▼
Prediction
        │
        ▼
Model Evaluation
        │
        ▼
Streamlit Web App
```

---

# 📈 Data Visualization

The project automatically generates:

- Pair Plot
- Heatmap
- Feature Distribution
- Box Plots
- Confusion Matrix
- Accuracy vs K Graph

All graphs are saved inside the **outputs/** folder.

---

# 🤖 Machine Learning Model

Algorithm Used:

**K-Nearest Neighbors (KNN)**

Why KNN?

- Easy to understand
- Good for small datasets
- High accuracy on Iris dataset
- Distance-based classification

---

# 📊 Evaluation Metrics

The project evaluates the model using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# ▶ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Iris-Flower-Classification-ML.git
```

Move inside project

```bash
cd Iris-Flower-Classification-ML
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Train Model

```bash
python train_model.py
```

This generates:

- iris.csv
- knn_model.pkl
- scaler.pkl
- Graphs

---

# ▶ Run Streamlit App

```bash
streamlit run app.py
```

---

# 📷 Screenshots

## Home Page

Add screenshot here

```
screenshots/home.png
```

---

## Prediction

Add screenshot here

```
screenshots/prediction.png
```

---

# 📈 Results

The KNN model achieves high accuracy on the Iris dataset.

Example Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

# 🔮 Future Improvements

- Hyperparameter Tuning
- GridSearchCV
- Cross Validation
- Random Forest
- Decision Tree
- Logistic Regression
- Support Vector Machine
- Model Deployment
- Docker Support


# ⭐ If you like this project

Please consider giving it a ⭐ on GitHub.