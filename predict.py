import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
#title
st.title("student performance predictor (advanced)")
#data set
data={
    "name":["A","B","C","D","E","F","G","H"],
    "DOB":["2000-05-10","2001-07-15","1999-08-20","2002-01-25",
           "2000-12-12","2001-03-30","1998-11-05","2002-06-18"],
    "Studyhours":[2,4,3,5,6,4,7,8],
    "Sleephours":[7,6,8,5,7,6,8,5],
    "previousScore":[55,65,60,70,75,68,80,85],
    "FinalScore":[58,68,63,73,78,70,83,88]
}
df=pd.DataFrame(data)
st.subheader("dataset preview")
st.dataframe(df)
#preprocessing
st.subheader("data preprocessing")
#covert dob to datetime
df["DOB"]=pd.to_datetime(df["DOB"])
#create age
current_year=2026
df["age"]=current_year - df["DOB"].dt.year
df["BirthMonth"]=df["DOB"].dt.month
st.write("missing values:")
st.write(df.isnull().sum())
st.write("Duplicate rows:", df.duplicated().sum())
df=df.drop_duplicates()
#drop unnecessary columns
df=df.drop(["name","DOB"],axis=1)
st.write("processed data:")
st.dataframe(df)
#feature enginering
st.subheader("Feature engineering")
df["Totaleffort"]=df["Studyhours"]+df["Sleephours"]
st.write("after feature engineering:")
st.dataframe(df)
#visual
st.subheader("data visualization")
fig1=plt.figure()
plt.scatter(df["Studyhours"],df["FinalScore"])
plt.xlabel("age")
plt.ylabel("Final score")
plt.title("age vs score")
st.pyplot(fig1)
fig2=plt.figure()
plt.scatter(df["Sleephours"],df["FinalScore"])
plt.xlabel("sleep")
plt.ylabel("Final score")
plt.title("sleep vs score")
st.pyplot(fig2)
fig3=plt.figure()
plt.hist(df["FinalScore"])
plt.title("score distribution")
st.pyplot(fig3)
st.subheader("Correlation matrix")
st.dataframe(df.corr())
st.subheader("outlier detection")
fig4=plt.figure()
plt.boxplot(df["FinalScore"])
plt.title("outliers in final score")
st.pyplot(fig4)
st.subheader("model training")
x=df.drop("FinalScore",axis=1)
y=df["FinalScore"]
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.3,random_state=42
)
model=LinearRegression()
model.fit(x_train,y_train)
st.success("model trained successfully")
st.subheader("model evaluation")
y_pred=model.predict(x_test)
#mae=mean_absolute_error(y_test,y_pred)
#mse=mean_squared_error(y_test,y_pred)
st.subheader("predict score")
study_hours=st.slider("study hours",0.0,12.0,5.0)
sleep_hours=st.slider("sleep hours",0.0,12.0,6.0)
previous_score=st.slider("previous score",0,100,60)
age=st.slider("age",15,30,20)
birth_month=st.slider("birth month",1,12,6)
if st.button("predict"):
    total_effort=study_hours+sleep_hours
    input_data=np.array([[study_hours,sleep_hours,previous_score,age,birth_month,total_effort]])
    prediction=model.predict(input_data)
    st.success(f"predicted score: {prediction[0]:.2f}")