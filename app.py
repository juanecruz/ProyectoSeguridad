from flask import Flask, request, render_template, url_for
import numpy as np
import time
from model import X_train, X_test, y_test, feature_names, victim_model, victim_mse, victim_r2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

stolen_X = []
stolen_y = []
attacker_model = None
last_attack_time = 0
attack_defense_enabled = False

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None, stats=None,
                           victim_mse=round(victim_mse, 2), victim_r2=round(victim_r2, 2),
                           feature_names=feature_names, blocked=False,
                           defense=attack_defense_enabled)

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(request.form.get(f"feature{i}", 0)) for i in range(5)] + [0.0] * 5
    prediction = victim_model.predict([features])[0]
    return render_template("index.html", prediction=round(prediction, 2), stats=None,
                           victim_mse=round(victim_mse, 2), victim_r2=round(victim_r2, 2),
                           feature_names=feature_names, blocked=False,
                           defense=attack_defense_enabled)

@app.route("/attack", methods=["POST"])
def attack():
    global stolen_X, stolen_y, attacker_model, last_attack_time, attack_defense_enabled
    attack_defense_enabled = request.form.get("defense") == "on"
    now = time.time()

    if attack_defense_enabled and now - last_attack_time < 10:
        return render_template("index.html", prediction=None, stats=None,
                               victim_mse=round(victim_mse, 2), victim_r2=round(victim_r2, 2),
                               feature_names=feature_names, blocked=True,
                               defense=True)

    last_attack_time = now
    np.random.seed(0)
    X_attack = np.random.uniform(X_train.min(), X_train.max(), size=(200, X_train.shape[1]))
    y_attack = victim_model.predict(X_attack)

    stolen_X = X_attack
    stolen_y = y_attack

    attacker_model = LinearRegression()
    attacker_model.fit(stolen_X, stolen_y)

    y_pred = attacker_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    class Stats:
        def __init__(self, n, mse, r2):
            self.n = n
            self.mse = mse
            self.r2 = r2

    return render_template("index.html", prediction=None,
                           stats=Stats(len(stolen_X), round(mse, 2), round(r2, 2)),
                           victim_mse=round(victim_mse, 2), victim_r2=round(victim_r2, 2),
                           feature_names=feature_names, blocked=False,
                           defense=attack_defense_enabled)

if __name__ == "__main__":
    app.run(debug=True)

