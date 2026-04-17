import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('data/clean/Edge Merged Data.csv')

predictors = [
    'Ht', 'Wt','40yd', '40yd_done', 'Vertical', 'Vertical_done', 'Bench', 'Bench_done', 'Broad Jump', 'Broad Jump_done', '3Cone', '3Cone_done','Shuttle', 'Shuttle_done', 'games_played', 'tackles', 'tackles_for_loss', 'sacks', 'forced_fumbles', 'arm_length', 'hand_size'
]

X = df[predictors]
y = np.log(df['5year_approx_value'] + 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=10, shuffle=True, random_state=1)
r2_scores = []
rmse_scores = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 25.0, 50.0, 75.0, 85.0, 90.0, 95.0, 100.0], cv=10)
    ridge_cv.fit(X_train, y_train)

    y_pred = ridge_cv.predict(X_test)

    r2_scores.append(r2_score(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

ridge_final = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 25.0, 50.0, 75.0, 85.0, 90.0, 95.0, 100.0], cv=10)
ridge_final.fit(X_scaled, y)

coef_df = pd.DataFrame({
    'Feature': predictors,
    'Coefficient': ridge_final.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

results = {
    "R2 Mean": np.mean(r2_scores),
    "R2 Std": np.std(r2_scores),
    "RMSE Mean": np.mean(rmse_scores),
    "RMSE Std": np.std(rmse_scores)
}

with open("model results/k-fold cv results/ridge_regression_10fold.txt", "w") as f:
    f.write("Ridge Regression with 10-Fold CV (Log-Transformed Response)\n")
    f.write("="*70 + "\n")
    f.write(f"Selected alpha: {ridge_final.alpha_:.6f}\n\n")
    for key, value in results.items():
        f.write(f"{key}: {value:.4f}\n")
    f.write("\nCoefficients:\n")
    f.write(coef_df.to_string(index=False))