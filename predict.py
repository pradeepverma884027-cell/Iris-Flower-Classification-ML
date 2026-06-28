"""
=========================================================
Prediction Module
Iris Flower Classification using Machine Learning
=========================================================
"""

import joblib
import numpy as np


class IrisPredictor:

    def __init__(self):

        # Load trained model
        self.model = joblib.load("model/knn_model.pkl")

        # Load scaler
        self.scaler = joblib.load("model/scaler.pkl")

        # Flower Names
        self.class_names = [
            "Setosa",
            "Versicolor",
            "Virginica"
        ]

    # ----------------------------------------
    # Predict Flower
    # ----------------------------------------

    def predict(
        self,
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ):

        sample = np.array([
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        ])

        # Scale Input
        sample = self.scaler.transform(sample)

        # Prediction
        prediction = self.model.predict(sample)[0]

        return self.class_names[prediction]

    # ----------------------------------------
    # Prediction Probability
    # ----------------------------------------

    def predict_probability(
        self,
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ):

        sample = np.array([
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        ])

        sample = self.scaler.transform(sample)

        probabilities = self.model.predict_proba(sample)[0]

        result = {}

        for i in range(len(self.class_names)):
            result[self.class_names[i]] = round(
                probabilities[i] * 100,
                2
            )

        return result


# ==========================================
# Standalone Testing
# ==========================================

if __name__ == "__main__":

    predictor = IrisPredictor()

    flower = predictor.predict(
        5.1,
        3.5,
        1.4,
        0.2
    )

    print("Predicted Flower :", flower)

    print("\nPrediction Probabilities")

    print(
        predictor.predict_probability(
            5.1,
            3.5,
            1.4,
            0.2
        )
    )