
# TASK-02
# Customer Segmentation using K-Means Clustering
# Prodigy Infotech

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# -----------------------------------------
# 1. Create Sample Customer Dataset
# -----------------------------------------

data = {
    "CustomerID": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    ],

    "Age": [
        19, 21, 20, 23, 31,
        22, 35, 40, 30, 25,
        46, 35, 58, 24, 37,
        29, 45, 32, 50, 27
    ],

    "AnnualIncome": [
        15, 16, 17, 18, 20,
        25, 30, 35, 40, 45,
        50, 55, 60, 65, 70,
        75, 80, 85, 90, 95
    ],

    "SpendingScore": [
        85, 80, 78, 75, 72,
        65, 60, 55, 50, 45,
        40, 35, 30, 25, 20,
        18, 15, 12, 10, 8
    ]
}


# Convert data into DataFrame
df = pd.DataFrame(data)


# -----------------------------------------
# 2. Display Dataset
# -----------------------------------------

print("CUSTOMER DATASET")
print("----------------")
print(df)


# -----------------------------------------
# 3. Select Features
# -----------------------------------------

X = df[["AnnualIncome", "SpendingScore"]]


# -----------------------------------------
# 4. Elbow Method
# -----------------------------------------

inertia = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    inertia.append(model.inertia_)


# Display Elbow Graph

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.grid(True)

plt.show()


# -----------------------------------------
# 5. Apply K-Means Clustering
# -----------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)


# -----------------------------------------
# 6. Display Cluster Centers
# -----------------------------------------

print("\nCLUSTER CENTERS")
print("----------------")

print(kmeans.cluster_centers_)


# -----------------------------------------
# 7. Display Customer Groups
# -----------------------------------------

print("\nCUSTOMER SEGMENTATION")
print("---------------------")

print(df)


# -----------------------------------------
# 8. Display Customers in Each Cluster
# -----------------------------------------

for cluster in sorted(df["Cluster"].unique()):

    print("\nCluster", cluster)

    customers = df[df["Cluster"] == cluster]

    print(
        customers[
            [
                "CustomerID",
                "Age",
                "AnnualIncome",
                "SpendingScore"
            ]
        ]
    )


# -----------------------------------------
# 9. Visualize Clusters
# -----------------------------------------

plt.figure(figsize=(10, 6))

colors = ["red", "blue", "green"]

for cluster in range(3):

    cluster_data = df[df["Cluster"] == cluster]

    plt.scatter(
        cluster_data["AnnualIncome"],
        cluster_data["SpendingScore"],
        s=100,
        color=colors[cluster],
        label=f"Cluster {cluster}"
    )


# Plot centroids

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    color="yellow",
    marker="*",
    edgecolor="black",
    label="Centroids"
)


plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.title("Customer Segmentation using K-Means")

plt.legend()

plt.grid(True)

plt.show()

