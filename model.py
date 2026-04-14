import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

attendance_data = pd.read_csv("data/dataset.csv")

X = attendance_data[attendance_data['Semester'] == 1]['SemOffset']
y = attendance_data[attendance_data['Semester'] == 1]['PastAttendance']

# X vs ln(y) looks to be pretty linear

X = pd.DataFrame(X)
y = pd.DataFrame(np.log(y))

model = LinearRegression()
model.fit(X, y)

print(r2_score(y, model.predict(X)))

plt.scatter(X, np.exp(y))
plt.plot(X, np.exp(model.predict(X)))
plt.show()
