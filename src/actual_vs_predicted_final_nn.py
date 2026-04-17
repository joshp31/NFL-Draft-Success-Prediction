import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/clean/Edge Merged Data.csv')

predictors = [
    'tackles_for_loss', '40yd', 'Wt', 'tackles', 'hand_size', 'Broad Jump_done', '3Cone'
]

X = df[predictors].values
y = df['5year_approx_value'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

nn_final = MLPRegressor(
    hidden_layer_sizes=(8, 4),
    max_iter=4000,
    alpha=0.001,
    random_state=1
)

nn_final.fit(X_scaled, y)

y_pred = nn_final.predict(X_scaled)

plt.figure(figsize=(7, 7))

plt.scatter(y, y_pred, alpha=0.6)

min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.xlabel("Actual 5-Year AV")
plt.ylabel("Predicted 5-Year AV")
plt.title("Neural Network: Actual vs Predicted 5-Year AV")

plt.tight_layout()
plt.savefig("model results/pred_vs_actual_nn.png", dpi=300, bbox_inches='tight')
plt.close()