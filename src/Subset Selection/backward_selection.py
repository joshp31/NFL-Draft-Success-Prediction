import pandas as pd
import statsmodels.api as sm
import numpy as np

def backward_selection(df, response, predictors):
    selected = predictors.copy()
    best_adj_r2 = -np.inf
    best_model = None

    while len(selected) > 0:
        results = []

        for predictor in selected:
            trial_predictors = selected.copy()
            trial_predictors.remove(predictor)

            X = sm.add_constant(df[trial_predictors])
            y = np.log(df[response] + 1)

            model = sm.OLS(y, X).fit()
            results.append((model.rsquared_adj, predictor, model))

        results.sort(reverse=True, key=lambda x: x[0])
        best_new_adj_r2, worst_predictor, best_new_model = results[0]

        if best_new_adj_r2 <= best_adj_r2:
            break

        best_adj_r2 = best_new_adj_r2
        best_model = best_new_model
        selected.remove(worst_predictor)

    with open("model results/subset selection results/backward_selection_summary.txt", "w") as f:
        f.write("Backward Selection Results\n")
        f.write("="*50 + "\n")
        f.write("Selected Variables:\n")
        f.write(", ".join(selected) + "\n\n")
        f.write(best_model.summary().as_text())

    return best_model, selected

if __name__ == "__main__":
    df = pd.read_csv('data/clean/Edge Merged Data.csv')
    predictors = [
        'Ht', 'Wt','40yd', '40yd_done', 'Vertical', 'Vertical_done', 'Bench', 'Bench_done', 'Broad Jump', 'Broad Jump_done', '3Cone', '3Cone_done', 'Shuttle', 'Shuttle_done', 'games_played', 'tackles', 'tackles_for_loss', 'sacks', 'forced_fumbles', 'arm_length', 'hand_size'
    ]

    model, selected_vars = backward_selection(df, '5year_approx_value', predictors)