# Attendance Prediction

A linear regression model that predicts event attendance and recommends a budget allocation based on historical data.

## How it works

The model takes past attendance data and fits a linear regression using two features — where in the semester an event falls (`SemOffset`) and which semester it is (fall or spring). It automatically tries different power transformations on the response variable to find the best fit, then uses that transformation for predictions.

## Usage

**Plot predictions against historical data:**
```
python model.py
```

**Predict attendance and get a budget recommendation:**
```
python model.py <sem_offset> <semester> <total_budget>
```

- `sem_offset` — week or offset within the semester
- `semester` — `0` for fall, `1` for spring
- `total_budget` — total budget to allocate across all events (integer)

Example:
```
python model.py 5 0 5000
```

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

Install with `pip install -r requirements.txt` (or just the packages above).

## Data

The model expects a CSV at `data/dataset.csv` with columns: `SemOffset`, `Semester`, `PastAttendance`.
