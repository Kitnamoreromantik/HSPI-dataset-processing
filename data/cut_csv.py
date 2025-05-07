import pandas as pd

df = pd.read_csv("data/prompts.csv")
df_head = df.head(3)

# Save to new file
df_head.to_csv("data/prompts_cut.csv", index=False)
