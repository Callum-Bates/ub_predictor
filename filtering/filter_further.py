# import pandas as pd

# # load your existing data
# df = pd.read_csv("/home/callum/projects/ub_predictor/data/processed/full_test.txt", sep="\t")

# # keep only what the pipeline needs and rename
# output = pd.DataFrame({
#     "protein_id"      : df["protein_id"],
#     "lysine_position" : df["position"].astype(int),
#     "ub"              : df["ub"]
# })

# # remove any rows where position is null
# output = output.dropna(subset=["lysine_position"])

# print(f"{len(output)} sites")
# print(f"{output['ub'].sum()} ubiquitinated")
# print(f"{(output['ub'] == 0).sum()} not ubiquitinated")
# print()
# print(output.head())

# # save
# output.to_csv("/home/callum/projects/ub_predictor/data/processed/full_test_ready.csv", index=False)




import pandas as pd
df = pd.read_csv('data/processed/full_test_ready.csv')
sample = df.groupby('ub').sample(250, random_state=42)  # 250 pos + 250 neg
sample.to_csv('data/raw/sites_sample_500.csv', index=False)
print(f'sample: {len(sample)} sites, {sample["protein_id"].nunique()} proteins')