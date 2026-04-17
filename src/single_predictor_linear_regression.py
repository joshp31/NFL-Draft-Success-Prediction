import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

def fit_small_model(df, predictors, response, output_dir):
    """
    Fits a small linear model with 1-2 predictors.
    Generates summary, residuals vs fitted plot, and scatter plot.
    
    Parameters:
        df (pd.DataFrame): input data
        predictors (list): list of 1-2 predictor column names
        response (str): response column name
        output_dir (str): directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    X = df[predictors]
    y = df[response]

    y_model = y

    X_const = sm.add_constant(X)
    model = sm.OLS(y_model, X_const).fit()

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(model.summary().as_text())

    # Residuals vs Fitted
    predictions = model.predict(X_const)
    residuals = y_model - predictions
    plt.figure(figsize=(6,4))
    plt.scatter(predictions, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_vs_fitted.png"))
    plt.close()

    # Scatter plot(s) with regression line
    for pred in predictors:
        plt.figure(figsize=(6,4))
        plt.scatter(df[pred], y, alpha=0.7, label="Actual")
        sorted_idx = df[pred].argsort()
        x_sorted = df[pred].iloc[sorted_idx]
        y_pred_sorted = model.predict(X_const).iloc[sorted_idx]
        plt.plot(x_sorted, y_pred_sorted, color='red', label="Linear fit")
        plt.xlabel(pred)
        plt.ylabel(response)
        plt.title(f"{pred} vs {response}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_{pred}.png"))
        plt.close()

    return model

def run_all_small_models(df, small_model_dict, base_output_dir="model results/single predictor linear regression results"):
    """
    Runs multiple small models based on a dictionary of predictors.
    
    Parameters:
        df (pd.DataFrame): input data
        small_model_dict (dict): keys = model name, values = list of predictors
        base_output_dir (str): base folder to save outputs
    """
    for model_name, predictors in small_model_dict.items():
        output_dir = os.path.join(base_output_dir, model_name)
        fit_small_model(df, predictors, '5year_approx_value', output_dir)

if __name__ == "__main__":
    df = pd.read_csv('data/clean/Edge Merged Data.csv')

    # Define small models (1-2 predictors)
    small_models = {
        "Ht": ["Ht"],
        "Wt": ["Wt"],
        "40yd": ["40yd", "40yd_done"],
        "Vertical": ["Vertical", "Vertical_done"],
        "Bench": ["Bench", "Bench_done"],
        "Broad Jump": ["Broad Jump", "Broad Jump_done"],
        "3Cone": ["3Cone", "3Cone_done"],
        "Shuttle": ["Shuttle", "Shuttle_done"],
        "games_played": ["games_played"],
        "tackles": ["tackles"],
        "tackles_for_loss": ["tackles_for_loss"],
        "sacks": ["sacks"],
        "forced_fumbles": ["forced_fumbles"],
        "arm_length": ["arm_length"],
        "hand_size": ["hand_size"]
    }

    run_all_small_models(df, small_models)