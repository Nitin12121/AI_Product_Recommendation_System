import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("data/ratings.csv")

user_product = ratings.pivot_table(
    index="user_id",
    columns="product_id",
    values="rating",
    fill_value=0
)

similarity = cosine_similarity(user_product)
similarity_df = pd.DataFrame(
    similarity,
    index=user_product.index,
    columns=user_product.index
)

print("User Similarity Matrix:")
print(similarity_df)