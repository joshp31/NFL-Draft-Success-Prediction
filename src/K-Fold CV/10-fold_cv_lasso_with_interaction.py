import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('data/clean/Edge Merged Data.csv')

df['speed_weight'] = df['Wt'] / df['40yd']

df['explosion'] = df['Vertical'] + df['Broad Jump']

df['power'] = df['Bench'] * df['Wt']

df['agility'] = 1 / (df['3Cone'] + df['Shuttle'])

df['production'] = (
    df['tackles'] +
    2 * df['tackles_for_loss'] +
    3 * df['sacks'] +
    2 * df['forced_fumbles']
)

df['length_tackles'] = df['arm_length'] * df['tackles']
df['length_tfl'] = df['arm_length'] * df['tackles_for_loss']

df['strength_control'] = df['Bench'] * df['arm_length']

predictors = [
    'Ht', 'Wt', 'arm_length', 'speed_weight', 'explosion', 'power', 'agility', 'production', 'length_tackles', 'length_tfl', 'strength_control'
]

X = df[predictors]
y = df['5year_approx_value']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=10, shuffle=True, random_state=1)

r2_scores = []
rmse_scores = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    lasso_cv = LassoCV(cv=10, random_state=1, max_iter=10000)
    lasso_cv.fit(X_train, y_train)

    y_pred = lasso_cv.predict(X_test)

    r2_scores.append(r2_score(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

results = {
    "R2 Mean": np.mean(r2_scores),
    "R2 Std": np.std(r2_scores),
    "RMSE Mean": np.mean(rmse_scores),
    "RMSE Std": np.std(rmse_scores)
}

lasso_final = LassoCV(cv=10, random_state=1, max_iter=10000)
lasso_final.fit(X_scaled, y)

coef_df = pd.DataFrame({
    'Feature': predictors,
    'Coefficient': lasso_final.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

with open("model results/k-fold cv results/lasso_interactions_10fold.txt", "w") as f:
    f.write("Lasso Regression (Interactions) - 10-Fold CV\n")
    f.write("="*70 + "\n")
    f.write(f"Selected alpha: {lasso_final.alpha_:.6f}\n\n")
    
    for key, value in results.items():
        f.write(f"{key}: {value:.4f}\n")
    
    f.write("\nFinal Model Metrics (Full Data):\n")
    y_pred_full = lasso_final.predict(X_scaled)
    f.write(f"R2: {r2_score(y, y_pred_full):.4f}\n")
    f.write(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_full)):.4f}\n")

    f.write("\nCoefficients:\n")
    f.write(coef_df.to_string(index=False))