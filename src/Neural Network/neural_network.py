import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

df = pd.read_csv('data/clean/Edge Merged Data.csv')

predictors = [
        'Ht', 'Wt','40yd', '40yd_done', 'Vertical', 'Vertical_done', 'Bench', 'Bench_done', 'Broad Jump', 'Broad Jump_done', '3Cone', '3Cone_done', 'Shuttle', 'Shuttle_done', 'games_played', 'tackles', 'tackles_for_loss', 'sacks', 'forced_fumbles', 'arm_length', 'hand_size'
    ]
X = df[predictors]
y = np.log(df['5year_approx_value'] + 1)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define neural network
nn = MLPRegressor(
    hidden_layer_sizes=(50,25), # To be tuned in train-test split phase
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=1
)

nn.fit(X_scaled, y)

y_pred = nn.predict(X_scaled)

r2 = r2_score(y, y_pred)

# Calculate adjusted R^2
n = X.shape[0]
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

with open("model results/neural network results/neural_network_summary.txt", "w") as f:
    f.write("Neural Network (MLP Regressor)\n")
    f.write("="*60 + "\n")
    f.write(f"Hidden Layers: (50, 25)\n")
    f.write(f"R^2: {r2:.4f}\n")
    f.write(f"Adjusted R^2: {adj_r2:.4f}\n")