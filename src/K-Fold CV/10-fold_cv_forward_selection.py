import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

def forward_selection_cv(X_train, y_train):
    selected = []
    remaining = list(X_train.columns)
    best_adj_r2 = -np.inf
    best_model = None

    while remaining:
        results = []
        for predictor in remaining:
            trial_predictors = selected + [predictor]
            X = sm.add_constant(X_train[trial_predictors])
            model = sm.OLS(y_train, X).fit()
            results.append((model.rsquared_adj, predictor, model))

        results.sort(reverse=True, key=lambda x: x[0])
        best_new_adj_r2, best_predictor, best_new_model = results[0]

        if best_new_adj_r2 <= best_adj_r2:
            break

        best_adj_r2 = best_new_adj_r2
        best_model = best_new_model
        selected.append(best_predictor)
        remaining.remove(best_predictor)

    return best_model, selected

def run_10fold_cv_forward_selection(df, response, predictors):
    X = df[predictors]
    y = np.log(df[response] + 1)

    kf = KFold(n_splits=10, shuffle=True, random_state=1)

    r2_scores = []
    rmse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model, selected_vars = forward_selection_cv(X_train, y_train)

        X_test_const = sm.add_constant(X_test[selected_vars])
        y_pred = model.predict(X_test_const)

        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    results = {
        "R2 Mean": np.mean(r2_scores),
        "R2 Std": np.std(r2_scores),
        "RMSE Mean": np.mean(rmse_scores),
        "RMSE Std": np.std(rmse_scores)
    }

    return results

def main():
    df = pd.read_csv('data/clean/Edge Merged Data.csv')
    predictors = [
        'Ht', 'Wt','40yd', '40yd_done', 'Vertical', 'Vertical_done', 'Bench', 'Bench_done', 'Broad Jump', 'Broad Jump_done', '3Cone', '3Cone_done', 'Shuttle', 'Shuttle_done', 'games_played', 'tackles', 'tackles_for_loss', 'sacks', 'forced_fumbles', 'arm_length', 'hand_size'
    ]

    results = run_10fold_cv_forward_selection(df, '5year_approx_value', predictors)

    with open("model results/k-fold cv results/forward_selection_10fold.txt", "w") as f:
        f.write("10-Fold Cross-Validation Results - Forward Selection\n")
        f.write("="*50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")

if __name__ == "__main__":
    main()