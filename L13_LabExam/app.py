from flask import Flask, request, render_template
import csv, random, os
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

ROLL = "23051707"
R = 7

CSV_FILE = "dataset.csv"
NUM_POINTS = 150

def generate_dataset():
    rows = []
    for i in range(NUM_POINTS):
        x = i + 1
        noise = random.uniform(-5, 5)
        y = (R * x) + noise - (R / 3)  
        rows.append((x, round(y, 4)))
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y"])
        writer.writerows(rows)
    return rows

def load_dataset():
    if not os.path.exists(CSV_FILE):
        generate_dataset()
    rows = []
    with open(CSV_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(float(row["X"])), float(row["Y"])))
    return rows

generate_dataset()

@app.route("/")
def home():
    rows = load_dataset()
    return render_template("home.html", rows=rows, roll=ROLL, r=R)

@app.route("/train")
def train():
    rows = load_dataset()
    X = np.array([r[0] for r in rows]).reshape(-1, 1)
    Y = np.array([r[1] for r in rows])
    model = LinearRegression().fit(X, Y)
    slope = round(model.coef_[0], 4)
    intercept = round(model.intercept_, 4)
    return render_template("train.html", slope=slope, intercept=intercept)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    x_val = None
    y_pred = None
    if request.method == "POST":
        x_val = float(request.form["x"])
        rows = load_dataset()
        X = np.array([r[0] for r in rows]).reshape(-1, 1)
        Y = np.array([r[1] for r in rows])
        model = LinearRegression().fit(X, Y)
        y_pred = round(model.predict([[x_val]])[0], 4)
    return render_template("predict.html", x_val=x_val, y_pred=y_pred)

if __name__ == "__main__":
    app.run(debug=True)