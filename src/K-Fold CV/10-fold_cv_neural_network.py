import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('data/clean/Edge Merged Data.csv')

# Use forward and backward selection predictors
predictors = ['tackles_for_loss', '40yd', 'Wt', 'tackles', 'hand_size', 'Broad Jump_done', '3Cone']

X = df[predictors].values
y = df['5year_approx_value'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=10, shuffle=True, random_state=1)
r2_scores = []
rmse_scores = []

for train_idx, test_idx in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    nn = MLPRegressor(hidden_layer_sizes=(8,4), max_iter=4000, alpha=0.001, random_state=1)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)

    r2_scores.append(r2_score(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

nn_final = MLPRegressor(hidden_layer_sizes=(8,4), max_iter=4000, alpha=0.001, random_state=1)
nn_final.fit(X_scaled, y)
y_pred_final = nn_final.predict(X_scaled)

r2_final = r2_score(y, y_pred_final)
n, p = X.shape
adj_r2_final = 1 - (1 - r2_final) * (n - 1) / (n - p - 1)
rmse_final = np.sqrt(mean_squared_error(y, y_pred_final))

with open("model results/k-fold cv results/neural_network_10fold.txt", "w") as f:
    f.write("Neural Network (MLP Regressor)\n")
    f.write("="*70 + "\n")
    f.write(f"Hidden Layers: (10, 5)\n\n")
    f.write(f"Regularization Strength (Alpha): {nn_final.alpha}\n\n")
    f.write("10-Fold CV Results:\n")
    f.write(f"R^2 Mean: {np.mean(r2_scores):.4f}, R^2 Std: {np.std(r2_scores):.4f}\n")
    f.write(f"RMSE Mean: {np.mean(rmse_scores):.4f}, RMSE Std: {np.std(rmse_scores):.4f}\n\n")
    f.write("Final Model Metrics (full data):\n")
    f.write(f"R^2: {r2_final:.4f}\n")
    f.write(f"Adjusted R^2: {adj_r2_final:.4f}\n")
    f.write(f"RMSE: {rmse_final:.4f}\n")