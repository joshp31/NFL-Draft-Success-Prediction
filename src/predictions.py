import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('data/clean/Edge Merged Data.csv')

predictors = ['tackles_for_loss', '40yd', 'Wt', 'tackles', 'hand_size', 'Broad Jump_done', '3Cone']

X = df[predictors]
y = df['5year_approx_value'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

nn_final = MLPRegressor(
    hidden_layer_sizes=(8,4),
    max_iter=4000,
    alpha=0.001,
    random_state=1
)

nn_final.fit(X_scaled, y)

df_new = pd.read_csv('data/clean/NFL Edge Combine Data2026.csv')

X_new = df_new[predictors].copy()

X_new = X_new.fillna(X_new.median())

X_new_scaled = scaler.transform(X_new)

df_new['predicted_AV'] = nn_final.predict(X_new_scaled)

df_plot = df_new.sort_values(by='predicted_AV', ascending=False)

plt.style.use('seaborn-v0_8-darkgrid')

plt.figure(figsize=(10, 10))

max_val = df_plot['predicted_AV'].max()

gap = max_val - df_plot['predicted_AV']
norm_gap = gap / gap.max()

colors = plt.cm.viridis(norm_gap)

bars = plt.barh(df_plot['Player'], df_plot['predicted_AV'], color=colors)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.02,
             bar.get_y() + bar.get_height()/2,
             f'{width:.1f}',
             va='center',
             fontsize=11,
             fontweight='bold')

plt.title("2026 EDGE Draft Model Rankings", fontsize=16, fontweight='bold')
plt.xlabel("Predicted 5-Year AV", fontsize=12, fontweight='bold')
plt.yticks(fontsize=11, fontweight='bold')
plt.xticks(fontsize=11, fontweight='bold')
plt.ylim(-0.7, len(df_plot) - 0.3)

plt.gca().invert_yaxis()

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("model results/predictions.png", dpi=300, bbox_inches='tight')
plt.close()