import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

attendance_data = pd.read_csv("data/dataset.csv")

X = attendance_data[['Semester', 'SemOffset']]
y = attendance_data['PastAttendance']

# Reference for main model 
model = LinearRegression()

def generate_best_fit(lam):
    # Instantiate experimental model
    experiment = LinearRegression()
    
    # Keep track of current r-squared
    r2 = 0

    # Fit experimenent to data based on current position in ladder of powers
    # Calculate r-squared for current lambda
    if lam != 0:
        experiment.fit(X, y ** lam)
        r2 = r2_score(y ** lam, experiment.predict(X))
    else:
        experiment.fit(X, np.log(y))
        r2 = r2_score(np.log(y), experiment.predict(X))

    # Display r-squared
    print(f'R-squared (lam = {lam}): {r2}')

    nxt = generate_best_fit(lam - 0.5) if lam != -1 else (0, 0)

    if r2 < nxt[1]:
        return (nxt[0], nxt[1])
    else:
        return (lam, r2)

lam, r2 = generate_best_fit(1)

print(f"Fitting model with transformation of lambda = {lam} for expected r-squared of {r2}.")

# Fit the model using the best transformation
model.fit(X, y ** lam if lam != 0 else np.log(y))

# Form results dataframe
results = X.join(pd.DataFrame(model.predict(X))).set_axis(['Semester', 'SemOffset', 'AttendanceBar'], axis=1)
results['AttendanceBar'] = results['AttendanceBar'] ** lam if lam != 0 else np.exp(results['AttendanceBar'])

# Plot fall semester values and predictions
plt.scatter(attendance_data[attendance_data['Semester'] == 0]['SemOffset'], attendance_data[attendance_data['Semester'] == 0]['PastAttendance'])
plt.plot(results[results['Semester'] == 0]['SemOffset'], results[results['Semester'] == 0]['AttendanceBar'])

# Plot spring semester values and predictionss
plt.scatter(attendance_data[attendance_data['Semester'] == 1]['SemOffset'], attendance_data[attendance_data['Semester'] == 1]['PastAttendance'])
plt.plot(results[results['Semester'] == 1]['SemOffset'], results[results['Semester'] == 1]['AttendanceBar'])

plt.show()
