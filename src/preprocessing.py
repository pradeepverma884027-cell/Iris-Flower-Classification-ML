"""
=========================================================
Preprocessing Module
Iris Flower Classification Project
=========================================================
"""

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:

    def __init__(self):

        self.iris = load_iris()

        self.scaler = StandardScaler()

    # ----------------------------------------
    # Load Dataset
    # ----------------------------------------

    def load_dataset(self):

        df = pd.DataFrame(
            self.iris.data,
            columns=self.iris.feature_names
        )

        df["target"] = self.iris.target

        df["species"] = df["target"].map({
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        })

        return df

    # ----------------------------------------
    # Features and Labels
    # ----------------------------------------

    def get_features_labels(self):

        X = self.iris.data

        y = self.iris.target

        return X, y

    # ----------------------------------------
    # Train Test Split
    # ----------------------------------------

    def split_data(
        self,
        X,
        y,
        test_size=0.20,
        random_state=42
    ):

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    # ----------------------------------------
    # Feature Scaling
    # ----------------------------------------

    def scale_data(
        self,
        X_train,
        X_test
    ):

        X_train_scaled = self.scaler.fit_transform(
            X_train
        )

        X_test_scaled = self.scaler.transform(
            X_test
        )

        return (
            X_train_scaled,
            X_test_scaled,
            self.scaler
        )

    # ----------------------------------------
    # Feature Names
    # ----------------------------------------

    def get_feature_names(self):

        return self.iris.feature_names

    # ----------------------------------------
    # Target Names
    # ----------------------------------------

    def get_target_names(self):

        return self.iris.target_names