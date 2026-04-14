from sklearn.linear_model import LinearRegression
import pandas as pd

data1={ 
    "area": [50,150,250,350,450,550],
    "price": [30,35,50,60,75,85]
}

df=pd.DataFrame(data1)
x=df[["area"]] #features go in double brackets
y=df["price"]

model1=LinearRegression()
model1.fit(x,y)

pred=model1.predict([[900]])
print("Value for 900 is : ", pred)
