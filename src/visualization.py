"""
=========================================================
Visualization Module
Iris Flower Classification Project
=========================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualizer:

    def __init__(self, dataframe):
        self.df = dataframe
        os.makedirs("outputs", exist_ok=True)

    def pair_plot(self):

        sns.pairplot(
            self.df,
            hue="species",
            palette="Set2"
        )

        plt.savefig(
            "outputs/pairplot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("✔ Pair Plot Saved")

    def correlation_heatmap(self):

        plt.figure(figsize=(8,6))

        correlation = self.df.drop(
            columns=["species"]
        ).corr(numeric_only=True)

        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            linewidths=0.5
        )

        plt.title("Feature Correlation Heatmap")

        plt.tight_layout()

        plt.savefig(
            "outputs/heatmap.png",
            dpi=300
        )

        plt.close()

        print("✔ Heatmap Saved")

    def feature_distribution(self):

        features = self.df.columns[:-2]

        for feature in features:

            plt.figure(figsize=(7,5))

            sns.histplot(
                data=self.df,
                x=feature,
                hue="species",
                kde=True,
                palette="Set2"
            )

            plt.title(f"Distribution of {feature}")

            filename = feature.replace(" ", "_")

            plt.tight_layout()

            plt.savefig(
                f"outputs/{filename}.png",
                dpi=300
            )

            plt.close()

        print("✔ Feature Distribution Plots Saved")

    def box_plot(self):

        features = self.df.columns[:-2]

        for feature in features:

            plt.figure(figsize=(7,5))

            sns.boxplot(
                data=self.df,
                x="species",
                y=feature,
                palette="Set2"
            )

            plt.title(f"{feature} Box Plot")

            filename = feature.replace(" ", "_")

            plt.tight_layout()

            plt.savefig(
                f"outputs/{filename}_boxplot.png",
                dpi=300
            )

            plt.close()

        print("✔ Box Plots Saved")

    def class_distribution(self):

        plt.figure(figsize=(6,5))

        sns.countplot(
            data=self.df,
            x="species",
            palette="Set2"
        )

        plt.title("Flower Class Distribution")

        plt.tight_layout()

        plt.savefig(
            "outputs/class_distribution.png",
            dpi=300
        )

        plt.close()

        print("✔ Class Distribution Plot Saved")

    def generate_all(self):

        print("\nGenerating Visualizations...\n")

        self.pair_plot()

        self.correlation_heatmap()

        self.feature_distribution()

        self.box_plot()

        self.class_distribution()

        print("\nAll Graphs Generated Successfully.")