import pandas as pd

df = pd.read_csv('data/raw/NFL Edge Combine Data 2026.csv')

df['Ht'] = df['Ht'].apply(lambda x: int(x.split('-')[0]) * 12 + int(x.split('-')[1]) if pd.notna(x) else None)

def parse_fraction(val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    if ' ' in val:
        whole, frac = val.split(' ')
        num, denom = frac.split('/')
        return float(whole) + float(num)/float(denom)
    elif '/' in val:
        num, denom = val.split('/')
        return float(num)/float(denom)
    else:
        return float(val)

df['hand_size'] = df['hand_size'].apply(parse_fraction)
df['arm_length'] = df['arm_length'].apply(parse_fraction)

per_game_cols = ["tackles", "tackles_for_loss", "sacks", "forced_fumbles"]
for col in per_game_cols:
    df[col] = df.apply(lambda row: row[col]/row["games_played"] if row["games_played"] > 0 else 0, axis=1)

combine_stats = ["40yd","Vertical", "Bench", "Broad Jump", "3Cone", "Shuttle"]

done_cols = ["40yd_done","Vertical_done","Bench_done","Broad Jump_done","3Cone_done","Shuttle_done"]
for col, combine_col in zip(done_cols, combine_stats):
    if col not in df.columns:
        df[col] = 0
    df[col] = df[combine_col].apply(lambda x: 1 if pd.notna(x) else 0)

df[combine_stats] = df[combine_stats].fillna(df[combine_stats].mean())

df.to_csv('data/clean/NFL Edge Combine Data2026.csv', index=False)