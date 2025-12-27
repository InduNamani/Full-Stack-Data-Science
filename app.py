import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


st.set_page_config("Linear Regression",layout="centered")
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")

st.markdown("""
            <div class="card">
            <h1>Linear Regression </h1>
            <p> Predict <b> Tip Amount </b> total Bill </b> using Linear regression..</p>
            </div>
            """,unsafe_allow_html=True)

@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

#Dataset Preview

st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())


#prepare the data
x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#Train Model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Metrics
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))  # ✅ Correct RMSE
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)

#visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")

fig, ax = plt.subplots()

# Scatter plot
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)

# Regression line
X = df[["total_bill"]]              # 2D input
X_scaled = scaler.transform(X)      # scale
y_pred = model.predict(X_scaled)    # predict

ax.plot(df["total_bill"], y_pred, color="red")

ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)



#Performance
st.markdown('<div class ="card">',unsafe_allow_html=True)
st.subheader("Model performance")
c1,c2=st.columns(2)
c1.metric("MAE",f"{mae: .2f}")
c2.metric("RMSE",f"{rmse: .2f}")
c3,c4=st.columns(2)
c3.metric("R2",f"{r2: .3f}")
c4.metric("adj R2",f"{adj_r2: .3f}")
st.markdown('</div>',unsafe_allow_html=True)


#m&c

st.markdown(f"""
            <div class="card">
            <h3>Model Intercept & Co-efficient<h3>
            <p><b>Co-efficient:</b>{model.coef_[0]:.3f}<br>
            <b>Intercept:</b>{model.intercept_:.3f}</p>
            </div>""",unsafe_allow_html=True)

#Predictions

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

tip = model.predict(scaler.transform([[bill]]))[0]

st.markdown(
    f'<div class="prediction-box">Predict Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

