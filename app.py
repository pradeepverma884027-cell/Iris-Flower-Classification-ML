"""
=========================================================
Iris Flower Classification Web Application
=========================================================
"""

import streamlit as st

from predict import IrisPredictor


# ---------------------------------------------
# Page Configuration
# ---------------------------------------------

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# ---------------------------------------------
# Load Predictor
# ---------------------------------------------

predictor = IrisPredictor()

# ---------------------------------------------
# Title
# ---------------------------------------------

st.title("🌸 Iris Flower Classification")
st.markdown("---")

st.write(
    """
This application predicts the species of an Iris flower using a
Machine Learning model trained with the **K-Nearest Neighbors (KNN)** algorithm.
"""
)

# ---------------------------------------------
# Sidebar
# ---------------------------------------------

st.sidebar.header("Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    4.0,
    8.0,
    5.1,
    0.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    2.0,
    4.5,
    3.5,
    0.1
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    1.0,
    7.0,
    1.4,
    0.1
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    0.1,
    2.5,
    0.2,
    0.1
)

# ---------------------------------------------
# Show Input
# ---------------------------------------------

st.subheader("Input Measurements")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Sepal Length",
        f"{sepal_length} cm"
    )

    st.metric(
        "Sepal Width",
        f"{sepal_width} cm"
    )

with col2:

    st.metric(
        "Petal Length",
        f"{petal_length} cm"
    )

    st.metric(
        "Petal Width",
        f"{petal_width} cm"
    )

st.markdown("---")

# ---------------------------------------------
# Prediction
# ---------------------------------------------

if st.button("Predict Flower"):

    prediction = predictor.predict(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    probabilities = predictor.predict_probability(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    st.success(
        f"Predicted Flower : **{prediction}**"
    )

    st.subheader("Prediction Probability")

    for flower, probability in probabilities.items():

        st.progress(probability / 100)

        st.write(
            f"{flower} : {probability}%"
        )

# ---------------------------------------------
# About Project
# ---------------------------------------------

st.markdown("---")

st.subheader("Project Information")

st.write("""
### Machine Learning Algorithm
- K-Nearest Neighbors (KNN)

### Dataset
- Iris Dataset
- 150 Samples
- 3 Flower Classes
- 4 Features

### Workflow

Dataset

↓

Preprocessing

↓

StandardScaler

↓

KNN Model

↓

Prediction

↓

Result
""")

# ---------------------------------------------
# Footer
# ---------------------------------------------

st.markdown("---")

st.caption(
    "Developed by Pradeep Kumar | AI & Machine Learning Internship Project"
)