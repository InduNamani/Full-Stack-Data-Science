import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

st.set_page_config("Lasso Regression - Salary Prediction", layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.markdown("""
            <div class="card">
            <h1>Lasso Regression</h1>
            <p>Predicting <b>Employee Salary</b> based on Years of Experience, Age, and Education Level using <b>L1 Regularization</b>.</p>
            </div>
            """, unsafe_allow_html=True)

@st.cache_data
def create_salary_data():
    np.random.seed(42)
    n = 200
    
    # Generate data
    experience = np.random.uniform(0, 20, n)
    age = 22 + experience + np.random.normal(0, 2, n)
    education = np.random.choice([1, 2, 3, 4], n)  # 1=High School, 2=Bachelor, 3=Master, 4=PhD
    
    # Salary formula with some noise
    salary = (
        30000 +  # Base salary
        3000 * experience +  # Experience impact
        500 * age +  # Age impact
        10000 * education +  # Education impact
        np.random.normal(0, 5000, n)  # Random noise
    )
    
    df = pd.DataFrame({
        'Experience': experience,
        'Age': age,
        'Education': education,
        'Salary': salary
    })
    
    return df

df = create_salary_data()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10))

st.markdown("**Dataset Info:**")
st.write(f"- Total Employees: {len(df)}")
st.write(f"- Average Salary: ${df['Salary'].mean():,.2f}")
st.write(f"- Education Levels: 1=High School, 2=Bachelor, 3=Master, 4=PhD")
st.markdown('</div>', unsafe_allow_html=True)

# Prepare data
X = df[['Experience', 'Age', 'Education']]
Y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Lasso model
alpha_value = st.sidebar.slider("Lasso Alpha (Regularization Strength)", 0.01, 10.0, 0.1)
model = Lasso(alpha=alpha_value)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# ---------------- VISUALIZATIONS ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Feature Selection by Lasso")

fig, ax = plt.subplots(figsize=(10, 5))
features = ['Experience', 'Age', 'Education']
coefficients = model.coef_
colors = ['red' if abs(c) < 0.01 else '#fa709a' for c in coefficients]
bars = ax.barh(features, coefficients, color=colors)
ax.set_xlabel("Impact on Salary (Coefficient)")
ax.set_title("Lasso Regression - Feature Importance (Red = Eliminated)")
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
st.pyplot(fig)

# Show eliminated features
zero_features = [f for f, c in zip(features, coefficients) if abs(c) < 0.01]
if zero_features:
    st.warning(f"⚠️ **Lasso eliminated these features**: {', '.join(zero_features)}")
    st.info("💡 Lasso removes less important features by setting their coefficients to zero!")
else:
    st.success("✅ **All features retained** - All are important for prediction!")

st.markdown('</div>', unsafe_allow_html=True)

# Experience vs Salary scatter
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("💼 Experience vs Salary Relationship")

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['Experience'], df['Salary'], c=df['Education'], 
                     cmap='plasma', alpha=0.6, s=100, edgecolors='black')
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary ($)")
ax.set_title("Experience vs Salary (Color = Education Level)")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Education Level')
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Actual vs Predicted
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎯 Actual vs Predicted Salaries")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.6, color='purple', edgecolors='black')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel("Actual Salary ($)")
ax.set_ylabel("Predicted Salary ($)")
ax.set_title("Model Predictions vs Reality")
ax.legend()
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Performance Metrics
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("RMSE", f"${rmse:,.0f}")
col3.metric("MAE", f"${mae:,.0f}")
col4.metric("MSE", f"${mse:,.0f}")

st.info(f"💡 **R² Score of {r2:.2%}** means the model explains {r2:.2%} of salary variation!")
st.markdown('</div>', unsafe_allow_html=True)

# Comparison with Ridge
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔍 Ridge vs Lasso - Key Differences")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Ridge Regression (L2)**")
    st.write("✓ Shrinks coefficients")
    st.write("✓ Keeps all features")
    st.write("✓ Good when all features matter")
    st.write("✓ Better for multicollinearity")

with col2:
    st.markdown("**Lasso Regression (L1)**")
    st.write("✓ Shrinks coefficients")
    st.write("✓ Can eliminate features (set to 0)")
    st.write("✓ Good for feature selection")
    st.write("✓ Simpler, more interpretable models")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Predict Employee Salary")

col1, col2, col3 = st.columns(3)

with col1:
    exp_input = st.slider("Years of Experience", 0, 20, 5)

with col2:
    age_input = st.slider("Age", 20, 65, 30)

with col3:
    edu_input = st.selectbox("Education Level", 
                             options=[1, 2, 3, 4],
                             format_func=lambda x: {1: "High School", 2: "Bachelor", 3: "Master", 4: "PhD"}[x])

# Prepare input
input_data = np.array([[exp_input, age_input, edu_input]])
input_scaled = scaler.transform(input_data)

# Predict
predicted_salary = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">💰 Predicted Salary: ${predicted_salary:,.2f}</div>',
    unsafe_allow_html=True
)

# Show breakdown
st.write("**Salary Breakdown:**")
if abs(model.coef_[0]) > 0.01:
    st.write(f"- Base contribution from Experience ({exp_input} years): ${model.coef_[0] * (exp_input - df['Experience'].mean()) / df['Experience'].std():,.2f}")
else:
    st.write(f"- Experience: ❌ Eliminated by Lasso")
    
if abs(model.coef_[1]) > 0.01:
    st.write(f"- Base contribution from Age ({age_input} years): ${model.coef_[1] * (age_input - df['Age'].mean()) / df['Age'].std():,.2f}")
else:
    st.write(f"- Age: ❌ Eliminated by Lasso")
    
if abs(model.coef_[2]) > 0.01:
    st.write(f"- Base contribution from Education (Level {edu_input}): ${model.coef_[2] * (edu_input - df['Education'].mean()) / df['Education'].std():,.2f}")
else:
    st.write(f"- Education: ❌ Eliminated by Lasso")

st.markdown('</div>', unsafe_allow_html=True)