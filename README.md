NFL EDGE Draft Prediction Model

Project Overview

This project develops predictive models to estimate NFL EDGE players’ Approximate Value (AV) over their first five seasons. This timeframe aligns with the typical rookie contract window, where teams expect early performance and return on investment. Approximate Value (AV) is used as the response variable as a general measure of player impact across seasons. While imperfect, it provides a standardized and widely available metric for comparing player performance. The models incorporate pre-draft data, including combine metrics, college production, and physical measurements, to evaluate how well measurable traits predict early NFL success. All statistical analyses are conducted using a 5% significance level.

Data

The dataset combines data from multiple sources:

https://www.sports-reference.com/cfb/ (EDGE college stats)

https://www.pro-football-reference.com/draft/2026-combine.htm (combine results, height, weight)

https://www.nfl.com/combine/tracker/live-results/ (hand size, arm length)

The final dataset includes EDGE prospects from 2008–2020, ensuring each player has at least five seasons of NFL performance data available for evaluation.

Data Preparation

Data from multiple sources (2008–2020) was merged into a unified EDGE-level dataset containing combine results, college production, and physical measurements. Manual position reclassification was performed to correct inaccurate positional labels. Indicator variables were created for combine participation, and missing values were imputed using mean values. Physical attributes such as height were standardized (converted from formats like 6-0 to total inches), and college statistics were converted to per-game averages.

Predictor Set:

The models use four groups of predictors: per-game college production statistics (games_played, tackles, tackles_for_loss, sacks, forced_fumbles), physical measurements (height, weight, arm_length, hand_size), and combine metrics (40yd, vertical, bench, broad_jump, 3cone, shuttle), along with binary indicator variables capturing whether a player participated in each combine event. In addition, college conference is included as a categorical predictor, grouped into Power 5, Group of 5, and FCS.

The response variable is 5-year Approximate Value (AV).

Initial Data Analysis

Simple linear regression (SLR) and multiple linear regression (MLR) models were initially explored. Both model types exhibited non-normal residuals, as observed in Q-Q plots. To address this, a Box-Cox transformation was applied to the response variable, suggesting a log transformation (λ ≈ 0.197). As a result, all regression, subset selection, and shrinkage models were fit using the log-transformed response.

Results from Initial Analysis

No train-test split was used in the initial models, as these analyses were purely exploratory. Across both transformed and untransformed linear regression and shrinkage models, the full linear model (excluding conference) achieved the highest Adjusted R² of 0.307. In single-predictor models, several variables showed statistically significant relationships with the response; however, all models had low R² values, indicating weak predictive power. No individual variable was a strong standalone predictor of 5-year AV. Forward and backward selection on the log-transformed response produced the same subset of predictors, resulting in an Adjusted R² of 0.260. Initial decision tree, random forest, and neural network models were also implemented, but were not optimized until the cross-validation stage.

10-fold Cross Validation

K-fold cross-validation (k = 10) was used to evaluate model performance for the most promising approaches. Models evaluated included transformed linear regression, forward and backward selection, ridge regression, lasso regression, random forest, and a neural network.

Results

The log transformation of the response improved performance for linear models but reduced performance for more flexible models such as random forests and neural networks. As a result, these models were evaluated without the log transformation during cross-validation.

The best-performing model from 10-fold cross-validation was a neural network, achieving a cross-validated Adjusted R² of 0.2697. This indicates that the model explains approximately 26.97% of the variation in 5-year AV, suggesting limited predictive power from the available features.

Conclusion

While the models do not strongly predict 5-year AV, the results highlight the limitations of using only pre-draft measurements and college production to estimate NFL success.

Key Findings

1. Arm length showed no statistically significant relationship with 5-year AV in single-variable models, suggesting limited standalone predictive value within this dataset. This result provides limited empirical support for the emphasis placed on arm length in some scouting evaluations. For example, EDGE prospect Reuben Bain Jr., who is often noted for shorter arm length measurements, may be cited in scouting discussions where this trait is weighted heavily. However, the model results suggest that this attribute alone is not a strong predictor of early NFL production. It is important to note that observations at the extreme lower end of arm length (such as Bain’s profile) are relatively rare in the dataset, which may limit the precision of inference in that range.

2. Pre-draft combine metrics and college production explain only a modest portion of early NFL career value, reinforcing the importance of contextual factors such as scheme fit, coaching, usage, and opportunity in player development.

Tools Used

- Python

- pandas

- numpy

- scikit-learn

- matplotlib

- statsmodels

- scipy

- os

File Structure
```
NFL-EDGE-Draft-Success-Prediction/
|
|── data/
| |── raw/ # Raw datasets before processing
| |── clean/ # Final processed datasets
|
|── model_results/ # Model outputs, evaluation results, and plots
| |── predictions  # Final predictions from best model (2026 EDGE class)
| |── pred_vs_actual_nn
|
|── src/ # Source code for all modeling and data processing
| |── data_processing  # Scripts to clean and transform raw data
| |── 2026_edge_data_processing
| |── single_predictor_linear_regression
| |── actual_vs_predicted_final_nn
| |── predictions  # Code used to generate final predictions and figures
|
|── .gitignore
|── README.md
└── requirements.txt
```
