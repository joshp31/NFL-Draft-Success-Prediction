import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

def run_10fold_cv(df):
    df_encoded = pd.get_dummies(df, columns=['conference'], drop_first=True)

    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    X = df_encoded.drop(columns=['player_id', 'Player', 'Pos', 'School', '5year_approx_value'])
    y = np.log(df_encoded['5year_approx_value'] + 1)

    kf = KFold(n_splits=10, shuffle=True, random_state=1)

    r2_scores = []
    rmse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        model = sm.OLS(y_train, X_train_const).fit()

        y_pred = model.predict(X_test_const)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        r2_scores.append(r2)
        rmse_scores.append(rmse)

    results = {
        "R2 Mean": np.mean(r2_scores),
        "R2 Std": np.std(r2_scores),
        "RMSE Mean": np.mean(rmse_scores),
        "RMSE Std": np.std(rmse_scores)
    }

    return results


def main():
    df = pd.read_csv('data/clean/Edge Merged Data.csv')

    results = run_10fold_cv(df)

    with open("model results/k-fold cv results/transformed_linear_regression_all_predictors_10fold.txt", "w") as f:
        f.write("10-Fold Cross-Validation Results\n")
        f.write("="*50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")

if __name__ == "__main__":
    main()