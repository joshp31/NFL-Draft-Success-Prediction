import pandas as pd
from scipy.stats import boxcox
import numpy as np


def main():
    df = pd.read_csv('data/clean/Edge Merged Data.csv')
    y = df['5year_approx_value']

    # Shift so y is greater than 0
    y_shifted = y + 1

    y_boxcox, lam = boxcox(y_shifted)

    with open(r"src\Box-Cox Transformation\box_cox_result.txt", "w") as f:
        f.write(f"Optimal lambda: {lam}\n")

if __name__ == "__main__":
    main()