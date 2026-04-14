import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/dataset.csv")

plt.plot(data.loc[data["Semester"] == 0, ["SemOffset"]], data.loc[data["Semester"] == 0, ["PastAttendance"]])
plt.plot(data.loc[data["Semester"] == 1, ["SemOffset"]], data.loc[data["Semester"] == 1, ["PastAttendance"]])

plt.show()
