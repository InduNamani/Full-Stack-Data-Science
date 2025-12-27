import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


st.set_page_config("Logistic Regression ",layout="centered")
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")

st.markdown("""
            <div class="card">
            <h1>Logistic regression</h1>
            <p>Based on age, job, balance, and contact type, the model predicts whether the customer will <b>subscribe to a <b>bank deposit....</p>
            </div>
            """,unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\indun\OneDrive\Documents\bank\bank.csv",sep=";")
df=load_data()

st.dataframe(df.head())

#changing to  numeric data
df['y']=df['y'].map({'yes':1,'no':0})

# for x also we have take the numerical values so 
cat_col= ['job', 'marital', 'education', 'default', 
                    'housing', 'loan', 'contact', 'month', 'poutcome']

df=pd.get_dummies(df,columns=cat_col,drop_first=True)

X=df.drop('y',axis=1)
# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Fill any missing values created by coercion
X = X.fillna(0)

Y=df['y']
X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)


X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=0.2)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=LogisticRegression(max_iter=100)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)

# ---------------- VISUALIZATION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Age vs Subscription Probability")

fig, ax = plt.subplots()

# Create age range
age_values = np.linspace(df["age"].min(), df["age"].max(), 100)

# Create empty matrix with same number of features
X_temp = np.zeros((100, X.shape[1]))

# Put age values in correct column index
age_index = list(X.columns).index("age")
X_temp[:, age_index] = age_values

# Scale data
X_scaled = scaler.transform(X_temp)

# Predict probabilities
y_prob = model.predict_proba(X_scaled)[:, 1]

# Plot
ax.plot(age_values, y_prob, color="red")
ax.set_xlabel("Age")
ax.set_ylabel("Probability of Subscription")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)




# Performance Card
st.markdown('<div class ="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

# Accuracy
c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy:.2f}")

# Confusion Matrix (display as text or table)
c2.text("Confusion Matrix:\n" + str(cm))

# Classification Report (display as text)
st.text("Classification Report:\n" + report)

st.markdown('</div>', unsafe_allow_html=True)



# Prediction Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Subscription Probability")

# Slider for Age
age = st.slider(
    "Customer Age",
    int(df['age'].min()),
    int(df['age'].max()),
    30
)

# Slider for Balance
balance = st.slider(
    "Account Balance",
    int(df['balance'].min()),
    int(df['balance'].max()),
    1000
)

# Slider for Duration (last contact in seconds)
duration = st.slider(
    "Last Contact Duration (seconds)",
    int(df['duration'].min()),
    int(df['duration'].max()),
    180
)

# Prepare input data (other features set to 0 for simplicity)
import numpy as np
input_data = np.array([[age, balance, duration] + [0]*(X_train.shape[1]-3)])  # fill other features with 0
input_data_scaled = scaler.transform(input_data)

# Predict probability and class
prob = model.predict_proba(input_data_scaled)[0,1]  # probability of 'Yes'
prediction = "Yes" if prob >= 0.5 else "No"

# Display prediction
st.markdown(
    f'<div class="prediction-box">Prediction: {prediction} <br> Probability: {prob:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True) 

