import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/clean/Edge Merged Data.csv')

predictors = [
        'Ht', 'Wt','40yd', '40yd_done', 'Vertical', 'Vertical_done', 'Bench', 'Bench_done', 'Broad Jump', 'Broad Jump_done', '3Cone', '3Cone_done', 'Shuttle', 'Shuttle_done', 'games_played', 'tackles', 'tackles_for_loss', 'sacks', 'forced_fumbles', 'arm_length', 'hand_size'
    ]
X = df[predictors]
y = np.log(df['5year_approx_value'] + 1)

tree = DecisionTreeRegressor(max_depth=4,random_state=1)
tree.fit(X, y)

y_pred = tree.predict(X)

r2 = r2_score(y, y_pred)

# Calculate adjusted R^2
n = X.shape[0]
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

importance = pd.DataFrame({
    'Feature': predictors,
    'Importance': tree.feature_importances_
}).sort_values(by='Importance', ascending=False)

with open("model results/decision tree results/decision_tree_summary.txt", "w") as f:
    f.write("Decision Tree Model - Log-Transformed 5-Year AV\n")
    f.write("="*60 + "\n")
    f.write(f"R^2: {r2:.4f}\n")
    f.write(f"Adjusted R^2: {adj_r2:.4f}\n\n")
    f.write("Feature Importances:\n")
    f.write(importance.to_string(index=False))

# Visualize and save the tree
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=predictors, filled=True, fontsize=10)
plt.title("Decision Tree for Log-Transformed 5-Year AV")
plt.tight_layout()
plt.savefig("model results/decision tree results/decision_tree_plot.png")
plt.close()