import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Naive Bayes Iris Classifier", layout="centered")

st.title("Naive Bayes (Gaussian & Bernoulli) using Iris Dataset")
st.write("🌸 Iris Flower Classification")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Train models
gnb = GaussianNB()
gnb.fit(x_train, y_train)

bnb = BernoulliNB()
bnb.fit(x_train, y_train)

# Sidebar inputs
st.sidebar.header("🌱 Input Flower Features")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean())
)
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean())
)
petal_length = st.sidebar.slider(
    "Petal Length (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean())
)
petal_width = st.sidebar.slider(
    "Petal Width (cm)", float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean())
)

model_choice = st.sidebar.selectbox(
    "Choose Naive Bayes Model",
    ("Gaussian Naive Bayes", "Bernoulli Naive Bayes")
)

# Prepare input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction
if model_choice == "Gaussian Naive Bayes":
    prediction = gnb.predict(input_data)[0]
    accuracy = accuracy_score(y_test, gnb.predict(x_test))
else:
    prediction = bnb.predict(input_data)[0]
    accuracy = accuracy_score(y_test, bnb.predict(x_test))

# Display results
st.subheader("🔍 Prediction Result")
st.success(f"Predicted Flower Type: **{target_names[prediction]}**")

st.subheader("📊 Model Performance")
st.write(f"Accuracy: **{accuracy * 100:.2f}%**")

# Show dataset
with st.expander("📁 View Iris Dataset"):
    df = pd.DataFrame(X, columns=feature_names)
    df["Target"] = y
    df["Flower Name"] = df["Target"].apply(lambda x: target_names[x])
    st.dataframe(df)