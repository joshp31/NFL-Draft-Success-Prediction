import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

def main():
    df = pd.read_csv('data/clean/Edge Merged Data.csv')

    # One-hot encode conference column
    df_encoded = pd.get_dummies(df, columns=['conference'], drop_first=True)

    # Convert boolean dummies to integers
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    X = df_encoded.drop(columns=['player_id', 'Player', 'Pos', 'School', '5year_approx_value'])
    y = df['5year_approx_value']

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

    with open("model results/linear regression results/linear_regression_all_predictors/summary.txt", "w") as f:
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
    plt.savefig("model results/linear regression results/linear_regression_all_predictors/residuals_vs_fitted.png")
    plt.close()

    # Actual vs. Predicted Plot
    plt.figure()
    plt.scatter(y, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.savefig("model results/linear regression results/linear_regression_all_predictors/actual_vs_predicted.png")
    plt.close()

    # Q-Q Plot
    plt.figure()
    sm.qqplot(residuals, line='45')
    plt.title("Q-Q Plot")
    plt.savefig("model results/linear regression results/linear_regression_all_predictors/qq_plot.png")
    plt.close()

if __name__ == "__main__":
    main()