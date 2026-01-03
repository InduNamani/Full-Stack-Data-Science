import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

st.set_page_config("Lasso Regression", layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.markdown("""
            <div class="card">
            <h1>Lasso Regression</h1>
            <p>Predicting <b>California House Prices</b> based on median income, house age, average rooms, and location features with <b>L1 regularization</b>.</p>
            </div>
            """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    return df

df = load_data()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# Prepare data
X = df.drop('Price', axis=1)
Y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Lasso model
model = Lasso(alpha=0.1)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# ---------------- VISUALIZATIONS ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Feature Importance (Coefficients)")

fig, ax = plt.subplots(figsize=(10, 6))
coefficients = pd.Series(model.coef_, index=X.columns).sort_values()
colors = ['red' if x == 0 else 'skyblue' for x in coefficients]
coefficients.plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel("Coefficient Value")
ax.set_title("Lasso Regression Coefficients (Red = Zeroed Out)")
st.pyplot(fig)

# Show which features were eliminated
zero_features = [f for f, c in zip(X.columns, model.coef_) if c == 0]
if zero_features:
    st.info(f"Features eliminated by Lasso: {', '.join(zero_features)}")
else:
    st.success("All features retained by Lasso")

st.markdown('</div>', unsafe_allow_html=True)

# Actual vs Predicted
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.5, color='purple')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Residuals Plot
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Residuals Distribution")

residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_pred, residuals, alpha=0.5, color='orange')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel("Predicted Price")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Performance Metrics
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("RMSE", f"{rmse:.3f}")
col3.metric("MAE", f"{mae:.3f}")
col4.metric("MSE", f"{mse:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict House Price")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.slider("Median Income (in $10k)", float(df['MedInc'].min()), float(df['MedInc'].max()), 3.0)
    house_age = st.slider("House Age (years)", float(df['HouseAge'].min()), float(df['HouseAge'].max()), 20.0)
    avg_rooms = st.slider("Avg Rooms", float(df['AveRooms'].min()), float(df['AveRooms'].max()), 5.0)
    avg_bedrooms = st.slider("Avg Bedrooms", float(df['AveBedrms'].min()), float(df['AveBedrms'].max()), 1.0)

with col2:
    population = st.slider("Population", float(df['Population'].min()), float(df['Population'].min()), 1000.0)
    avg_occup = st.slider("Avg Occupancy", float(df['AveOccup'].min()), float(df['AveOccup'].max()), 3.0)
    latitude = st.slider("Latitude", float(df['Latitude'].min()), float(df['Latitude'].max()), 35.0)
    longitude = st.slider("Longitude", float(df['Longitude'].min()), float(df['Longitude'].max()), -119.0)

# Prepare input
input_data = np.array([[med_inc, house_age, avg_rooms, avg_bedrooms, population, avg_occup, latitude, longitude]])
input_scaled = scaler.transform(input_data)

# Predict
predicted_price = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted House Price: ${predicted_price * 100000:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)