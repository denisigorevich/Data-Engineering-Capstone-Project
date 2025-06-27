import pandas as pd

df = pd.read_csv("../data/vehicles_cleaned.csv", on_bad_lines='skip', engine='python')
df.to_csv("../data/vehicles_cleaned_clean.csv", index=False, quoting=1)
