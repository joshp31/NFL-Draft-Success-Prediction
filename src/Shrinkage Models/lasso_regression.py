import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

df = pd.read_csv('data/clean/Edge Merged Data.csv')

predictors = [
        'Ht', 'Wt','40yd', '40yd_done', 'Vertical', 'Vertical_done', 'Bench', 'Bench_done', 'Broad Jump', 'Broad Jump_done', '3Cone', '3Cone_done', 'Shuttle', 'Shuttle_done', 'games_played', 'tackles', 'tackles_for_loss', 'sacks', 'forced_fumbles', 'arm_length', 'hand_size'
    ]
X = df[predictors]
y = np.log(df['5year_approx_value'] + 1)

# Standardize X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1, random_state=1)
lasso.fit(X_scaled, y)

y_pred = lasso.predict(X_scaled)

r2 = r2_score(y, y_pred)

# Calculate adjusted R^2
n = X.shape[0]
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Coefficients
coef_df = pd.DataFrame({
    'Feature': predictors,
    'Coefficient': lasso.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

with open("model results/shrinkage results/lasso_summary.txt", "w") as f:
    f.write("Lasso Regression (Log-Transformed Response)\n")
    f.write("="*60 + "\n")
    f.write(f"R^2: {r2:.4f}\n")
    f.write(f"Adjusted R^2: {adj_r2:.4f}\n\n")
    f.write("Coefficients:\n")
    f.write(coef_df.to_string(index=False))