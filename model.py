import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

attendance_data = pd.read_csv("data/dataset.csv")

X = attendance_data[['Semester', 'SemOffset']]
y = attendance_data['PastAttendance']

# Create and fit model on X vs. ln(y)
model = LinearRegression()
model.fit(X, np.log(y))

# Display r-squared
print(f'R-squared: {r2_score(y, model.predict(X))}')

# Form results dataframe
results = X.join(pd.DataFrame(model.predict(X))).set_axis(['Semester', 'SemOffset', 'AttendanceBar'], axis=1)
results['AttendanceBar'] = np.exp(results['AttendanceBar'])

# Plot fall semester values and predictions
plt.scatter(attendance_data[attendance_data['Semester'] == 0]['SemOffset'], attendance_data[attendance_data['Semester'] == 0]['PastAttendance'])
plt.plot(results[results['Semester'] == 0]['SemOffset'], results[results['Semester'] == 0]['AttendanceBar'])

# Plot spring semester values and predictionss
plt.scatter(attendance_data[attendance_data['Semester'] == 1]['SemOffset'], attendance_data[attendance_data['Semester'] == 1]['PastAttendance'])
plt.plot(results[results['Semester'] == 1]['SemOffset'], results[results['Semester'] == 1]['AttendanceBar'])

plt.show()
