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
            <h1>Logistic Regression</h1>
            <p>Based on age, job, balance, and contact type, the model predicts whether the customer will <b>subscribe to a <b>bank deposit....</p>
            </div>
            """,unsafe_allow_html=True)

import pandas as pd

columns = [
    'age','job','marital','education','default',
    'balance','housing','loan','contact','day',
    'month','duration','campaign','pdays','previous',
    'poutcome','y'
]

df = pd.read_csv("bank.csv", header=None, names=columns)


st.dataframe(df.head())


df['y'] = df['y'].map({'yes':1, 'no':0})  # now this will work


# for x also we have take the numerical values so 
cat_col= ['job', 'marital', 'education', 'default', 
                    'housing', 'loan', 'contact', 'month', 'poutcome']

df=pd.get_dummies(df,columns=cat_col,drop_first=True)

X=df.drop('y',axis=1)
Y=df['y']

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=0.2)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
fig, ax = plt.subplots()



# ------------------ Visualization ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Age vs Probability of Subscription")

fig, ax = plt.subplots()



# Create age range
x_values = np.linspace(df["age"].min(), df["age"].max(), 100)

# Create base input using mean of each feature
X_plot = np.tile(X_train.mean(axis=0), (100, 1))

# Find index of 'age' column
age_index = list(X.columns).index("age")

# Replace only the age column
X_plot[:, age_index] = x_values

# Predict probabilities
y_prob = model.predict_proba(X_plot)[:, 1]

# Scatter actual data
ax.scatter(df["age"], df["y"], alpha=0.3, label="Actual Data")

# Plot logistic curve
ax.plot(x_values, y_prob, color="red", linewidth=2, label="Logistic Curve")

ax.set_xlabel("Age")
ax.set_ylabel("Probability of Subscription")
ax.legend()

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.2f}")

with col2:
    st.text("Confusion Matrix")
    st.text(cm)

st.subheader("Classification Report")
st.text(report)

