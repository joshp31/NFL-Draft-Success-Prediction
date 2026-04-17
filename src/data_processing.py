import pandas as pd

def main():
    drafted_combine_stats_df = pd.read_csv('data/raw/NFL Combine Data 2008-2020 (Drafted).csv')
    drafted_edge_combine_stats_df = pd.read_csv('data/raw/Edge Stats and Other.csv')

    edge_merged_df = pd.merge(drafted_combine_stats_df, drafted_edge_combine_stats_df, on='player_id', how='inner')

    edge_merged_df_cleaned = edge_merged_df.drop(columns=["College", "Drafted (tm/rnd/yr)", "Player-additional", "Player_y", "School_y"])
    
    edge_merged_df_cleaned = edge_merged_df_cleaned.rename(columns={
        "Player_x": "Player",
        "School_x": "School"
    })

    edge_merged_df_cleaned['Ht'] = edge_merged_df_cleaned['Ht'].apply(
        lambda x: int(x.split('-')[0]) * 12 + int(x.split('-')[1])
    )

    # Convert college stats to per-game values (without renaming columns)
    per_game_cols = ["tackles", "tackles_for_loss", "sacks", "forced_fumbles"]

    for col in per_game_cols:
        edge_merged_df_cleaned[col] = edge_merged_df_cleaned[col] / edge_merged_df_cleaned["games_played"]

    # Create 0/1 indicator columns for missing combine stats
    combine_stats_cols = ["40yd","Vertical", "Bench", "Broad Jump", "3Cone", "Shuttle"]
    for col in combine_stats_cols:
        edge_merged_df_cleaned[col + "_done"] = edge_merged_df_cleaned[col].notna().astype(int)

    combine_stats = ["40yd","Vertical", "Bench", "Broad Jump", "3Cone", "Shuttle"]
    
    # Fill missing values with column mean
    for col in combine_stats:
        edge_merged_df_cleaned[col].fillna(edge_merged_df_cleaned[col].mean(), inplace=True)
    
    edge_merged_df_cleaned.to_csv('data/clean/Edge Merged Data.csv', index=False)

    print(edge_merged_df_cleaned.shape)
    print(edge_merged_df_cleaned.isna().sum())

if __name__ == "__main__":
    main()