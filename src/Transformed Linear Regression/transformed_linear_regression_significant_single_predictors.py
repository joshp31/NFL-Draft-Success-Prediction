import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

def main():
    df = pd.read_csv('data/clean/Edge Merged Data.csv')

    columns_to_keep = ['3Cone', '3Cone_done', '40yd', 'Broad Jump', 'Broad Jump_done', 'forced_fumbles', 'games_played', 'hand_size', 'sacks', 'Shuttle', 'Shuttle_done', 'tackles', 'tackles_for_loss', 'Vertical', 'Vertical_done', '5year_approx_value']

    X = df[columns_to_keep].drop(columns=['5year_approx_value'])
    y = np.log(df['5year_approx_value'] + 1)

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    # Make plots and print summary/VIF info
    X_no_const = X.drop(columns=["const"])

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_no_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_no_const.values, i)
        for i in range(X_no_const.shape[1])
    ]

    with open("model results/transformed linear regression results/linear_regression_significant_single_predictors/summary.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("VIF (Variance Inflation Factors)\n")
        f.write("="*80 + "\n\n")
        f.write(vif_data.to_string(index=False))

    predictions = model.predict(X)
    residuals = y - predictions

    # Residuals vs. Fitted
    plt.figure()
    plt.scatter(predictions, residuals)
    plt.axhline(0)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.savefig("model results/transformed linear regression results/linear_regression_significant_single_predictors/residuals_vs_fitted.png")
    plt.close()

    predictions_exp = np.exp(predictions) - 1
    y_exp = np.exp(y) - 1

    # Actual vs. Predicted
    plt.figure()
    plt.scatter(y_exp, predictions_exp)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.savefig("model results/transformed linear regression results/linear_regression_significant_single_predictors/actual_vs_predicted.png")
    plt.close()

    # Q-Q Plot
    plt.figure()
    sm.qqplot(residuals, line='45')
    plt.title("Q-Q Plot")
    plt.savefig("model results/transformed linear regression results/linear_regression_significant_single_predictors/qq_plot.png")
    plt.close()

if __name__ == "__main__":
    main()