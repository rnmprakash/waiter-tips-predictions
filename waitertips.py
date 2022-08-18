import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/tips.csv")
print(data.head())

figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "day", trendline="ols")
figure.show()

figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "sex", trendline="ols")
figure.show()

figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "time", trendline="ols")
figure.show()

figure = px.pie(data, 
             values='tip', 
             names='day',hole = 0.5)
figure.show()

figure = px.pie(data, 
             values='tip', 
             names='sex',hole = 0.5)
figure.show()

figure = px.pie(data, 
             values='tip', 
             names='smoker',hole = 0.5)
figure.show()

figure = px.pie(data, 
             values='tip', 
             names='time',hole = 0.5)
figure.show()

data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
data.head()

x = np.array(data[["total_bill", "sex", "smoker", "day", 
                   "time", "size"]])
y = np.array(data["tip"])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)
                                                
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)

features = np.array([[24.50, 1, 0, 0, 1, 4]])
model.predict(features)
