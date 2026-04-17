import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df = pd.read_csv('data/clean/Edge Merged Data.csv')

# Use forward and backward selection predictors
predictors = ['tackles_for_loss', '40yd', 'Wt', 'tackles', 'hand_size', 'Broad Jump_done', '3Cone']

X = df[predictors].values
y = df['5year_approx_value'].values

kf = KFold(n_splits=10, shuffle=True, random_state=1)

r2_scores = []
rmse_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    r2_scores.append(r2_score(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

results = {
    "R2 Mean": np.mean(r2_scores),
    "R2 Std": np.std(r2_scores),
    "RMSE Mean": np.mean(rmse_scores),
    "RMSE Std": np.std(rmse_scores)
}

rf_final = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
rf_final.fit(X, y)
importance = pd.DataFrame({
    'Feature': predictors,
    'Importance': rf_final.feature_importances_
}).sort_values(by='Importance', ascending=False)

with open("model results/k-fold cv results/random_forest_10fold.txt", "w") as f:
    f.write("Random Forest with 10-Fold CV\n")
    f.write("="*70 + "\n")
    for key, value in results.items():
        f.write(f"{key}: {value:.4f}\n")
    f.write("\nFeature Importances:\n")
    f.write(importance.to_string(index=False))