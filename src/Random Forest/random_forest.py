import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/clean/Edge Merged Data.csv')

predictors = [
        'Ht', 'Wt','40yd', '40yd_done', 'Vertical', 'Vertical_done', 'Bench', 'Bench_done', 'Broad Jump', 'Broad Jump_done', '3Cone', '3Cone_done', 'Shuttle', 'Shuttle_done', 'games_played', 'tackles', 'tackles_for_loss', 'sacks', 'forced_fumbles', 'arm_length', 'hand_size'
    ]
X = df[predictors]
y = np.log(df['5year_approx_value'] + 1)

# Fit Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=4,
    random_state=1
)

rf.fit(X, y)

y_pred = rf.predict(X)

r2 = r2_score(y, y_pred)

# Calculate adjusted R^2
n = X.shape[0]
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

importance = pd.DataFrame({
    'Feature': predictors,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

with open("model results/random forest results/random_forest_summary.txt", "w") as f:
    f.write("Random Forest Model - Log-Transformed 5-Year AV\n")
    f.write("="*60 + "\n")
    f.write(f"R^2: {r2:.4f}\n")
    f.write(f"Adjusted R^2: {adj_r2:.4f}\n\n")
    f.write("Feature Importances:\n")
    f.write(importance.to_string(index=False))

plt.figure(figsize=(10,6))
plt.barh(importance['Feature'], importance['Importance'])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("model results/random forest results/random_forest_summary.png")
plt.close()