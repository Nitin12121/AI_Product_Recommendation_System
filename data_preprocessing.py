import pandas as pd

users = pd.read_csv("data/users.csv")
products = pd.read_csv("data/products.csv")
ratings = pd.read_csv("data/ratings.csv")

data = ratings.merge(users, on="user_id").merge(products, on="product_id")

print("Merged Dataset:")
print(data.head())