"""
=========================================================
Iris Flower Classification using Machine Learning
Algorithm : K-Nearest Neighbors (KNN)
=========================================================
"""

# ==========================
# Import Libraries
# ==========================

from src.evaluation import ModelEvaluator
from src.visualization import DataVisualizer
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ==========================
# Create Required Folders
# ==========================

os.makedirs("dataset", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==========================
# Load Dataset
# ==========================

print("=" * 60)
print("Loading Iris Dataset...")
print("=" * 60)

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target
df["species"] = df["target"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

visualizer = DataVisualizer(df)
visualizer.generate_all()

# Save dataset as CSV
df.to_csv("dataset/iris.csv", index=False)

print("\nDataset Loaded Successfully.\n")

# ==========================
# Basic Dataset Information
# ==========================

print("=" * 60)
print("Dataset Information")
print("=" * 60)

print(df.head())

print("\nShape :", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nClass Distribution")
print(df["species"].value_counts())

# ==========================
# Features and Labels
# ==========================

X = iris.data
y = iris.target

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================
# Model Creation
# ==========================

model = KNeighborsClassifier(
    n_neighbors=5
)

# ==========================
# Model Training
# ==========================

print("\nTraining Model...")

model.fit(
    X_train,
    y_train
)

print("Training Completed.")

# ==========================
# Prediction
# ==========================

y_pred = model.predict(X_test)

evaluator = ModelEvaluator(
    iris.target_names
)

results = evaluator.evaluate(
    y_test,
    y_pred
)

# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy : {:.2f}%".format(
    accuracy * 100
))

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix\n")

print(cm)

# ==========================
# Save Model
# ==========================

joblib.dump(
    model,
    "model/knn_model.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

print("\nModel Saved Successfully.")

# ==========================
# Confusion Matrix Plot
# ==========================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion Matrix Saved.")

# ==========================
# Accuracy vs K Graph
# ==========================

k_values = range(1, 21)

scores = []

for k in k_values:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(
        X_train,
        y_train
    )

    pred = knn.predict(
        X_test
    )

    scores.append(
        accuracy_score(
            y_test,
            pred
        )
    )

plt.figure(figsize=(8,5))

plt.plot(
    k_values,
    scores,
    marker="o"
)

plt.title("Accuracy vs K")

plt.xlabel("Number of Neighbors (K)")

plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig(
    "outputs/accuracy_vs_k.png",
    dpi=300
)

plt.close()

print("Accuracy vs K Graph Saved.")

print("\n")
print("=" * 60)
print("Project Completed Successfully")
print("=" * 60)
print("Generated Files:")
print("✔ dataset/iris.csv")
print("✔ model/knn_model.pkl")
print("✔ model/scaler.pkl")
print("✔ outputs/confusion_matrix.png")
print("✔ outputs/accuracy_vs_k.png")
print("=" * 60)