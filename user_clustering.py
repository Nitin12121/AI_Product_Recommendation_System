import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ratings = pd.read_csv("data/ratings.csv")

user_matrix = ratings.pivot_table(
    index="user_id",
    columns="product_id",
    values="rating",
    fill_value=0
)

kmeans = KMeans(n_clusters=2, random_state=42)
user_matrix["cluster"] = kmeans.fit_predict(user_matrix)

print("User Clusters:")
print(user_matrix)

# Visualization
plt.scatter(user_matrix.iloc[:, 0], user_matrix.iloc[:, 1])
plt.title("User Clustering")
plt.xlabel("Product 1 Rating")
plt.ylabel("Product 2 Rating")
plt.savefig("outputs/clusters.png")
plt.show()