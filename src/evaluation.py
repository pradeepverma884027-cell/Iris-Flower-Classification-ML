"""
=========================================================
Evaluation Module
Iris Flower Classification Project
=========================================================
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class ModelEvaluator:

    def __init__(self, target_names):

        self.target_names = target_names

        os.makedirs("outputs", exist_ok=True)

    # ---------------------------------------
    # Accuracy
    # ---------------------------------------

    def accuracy(self, y_true, y_pred):

        return accuracy_score(
            y_true,
            y_pred
        )

    # ---------------------------------------
    # Precision
    # ---------------------------------------

    def precision(self, y_true, y_pred):

        return precision_score(
            y_true,
            y_pred,
            average="weighted"
        )

    # ---------------------------------------
    # Recall
    # ---------------------------------------

    def recall(self, y_true, y_pred):

        return recall_score(
            y_true,
            y_pred,
            average="weighted"
        )

    # ---------------------------------------
    # F1 Score
    # ---------------------------------------

    def f1(self, y_true, y_pred):

        return f1_score(
            y_true,
            y_pred,
            average="weighted"
        )

    # ---------------------------------------
    # Classification Report
    # ---------------------------------------

    def report(self, y_true, y_pred):

        return classification_report(
            y_true,
            y_pred,
            target_names=self.target_names
        )

    # ---------------------------------------
    # Confusion Matrix
    # ---------------------------------------

    def matrix(self, y_true, y_pred):

        return confusion_matrix(
            y_true,
            y_pred
        )

    # ---------------------------------------
    # Save Confusion Matrix
    # ---------------------------------------

    def save_confusion_matrix(
        self,
        y_true,
        y_pred
    ):

        cm = self.matrix(
            y_true,
            y_pred
        )

        plt.figure(figsize=(6,5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.target_names,
            yticklabels=self.target_names
        )

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(
            "outputs/confusion_matrix.png",
            dpi=300
        )

        plt.close()

        print("✔ Confusion Matrix Saved")

    # ---------------------------------------
    # Complete Summary
    # ---------------------------------------

    def evaluate(
        self,
        y_true,
        y_pred
    ):

        acc = self.accuracy(
            y_true,
            y_pred
        )

        pre = self.precision(
            y_true,
            y_pred
        )

        rec = self.recall(
            y_true,
            y_pred
        )

        f1 = self.f1(
            y_true,
            y_pred
        )

        print("\n")
        print("="*60)
        print("MODEL PERFORMANCE")
        print("="*60)

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {pre:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        print("\nClassification Report\n")

        print(
            self.report(
                y_true,
                y_pred
            )
        )

        self.save_confusion_matrix(
            y_true,
            y_pred
        )

        print("="*60)

        return {
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1_score": f1
        }