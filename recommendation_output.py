import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("data/ratings.csv")
products = pd.read_csv("data/products.csv")

user_product = ratings.pivot_table(
    index="user_id",
    columns="product_id",
    values="rating",
    fill_value=0
)

similarity = cosine_similarity(user_product)
similarity_df = pd.DataFrame(similarity, index=user_product.index, columns=user_product.index)

def recommend(user_id, top_n=3):
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:]
    weighted_scores = user_product.loc[similar_users.index].T.dot(similar_users)
    recommendations = weighted_scores.sort_values(ascending=False).head(top_n)
    
    return products[products["product_id"].isin(recommendations.index)]

result = recommend(user_id=2)

print("Recommended Products:")
print(result)

result.to_csv("outputs/recommendations.csv", index=False)