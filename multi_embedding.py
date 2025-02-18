import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance
from qdrant_client.http.models import VectorParams
from qdrant_client.http.models import PointStruct

# Sample dataset (subset of your sample data)
data = [
    {"ID": 1001, "Name": "vinodkumar nallamogla", "Age": 34},
    {"ID": 1002, "Name": "vinod katari", "Age": 32},
    {"ID": 1003, "Name": "vinod yerra", "Age": 30},
    {"ID": 1004, "Name": "bharath moti", "Age": 25},
    {"ID": 1005, "Name": "bharath motikayala", "Age": 26},
    {"ID": 1006, "Name": "ramesh katari", "Age": 29},
    {"ID": 1007, "Name": "ramesh dugasani", "Age": 30},
    {"ID": 1008, "Name": "satish dugasasni", "Age": 38},
    {"ID": 1009, "Name": "venu yerra", "Age": 42},
    {"ID": 1010, "Name": "sasi dugasani", "Age": 36},
]

# Create a DataFrame
df = pd.DataFrame(data)

# Number of records (we'll use this for the length of the one-hot vector)
n_records = len(df)

# 1. Generate Sparse Vectors (One-Hot Encoding)
def generate_one_hot(index, length):
    vec = [0.0] * length
    vec[index] = 1.0
    return vec

df['sparse_vector'] = [generate_one_hot(i, n_records) for i in range(n_records)]

# 2. Generate Zero Vectors (All Zeros)
df['zero_vector'] = [[0.0] * n_records for _ in range(n_records)]

#3. Generate Custom Vectors using Uniform Distribution
#    Let's generate a 10-dimensional vector with values from U(0,1)
df['uniform_vector'] = [np.random.uniform(0, 1, 10).round(2).tolist() for _ in range(n_records)]

# 4. Generate Custom Vectors using Normal Distribution
#    For example, with mean=0.5 and std=0.1, 10-dimensional
df['normal_vector'] = [np.random.normal(0.5, 0.1, 10).round(2).tolist() for _ in range(n_records)]

client = QdrantClient(
    url="https://64c9b924-a747-4c21-874f-f6901dc0431e.us-east-1-0.aws.cloud.qdrant.io:6333",  # e.g., "your-cluster-1234.us-east1.qdrant.io"
    api_key="vgywMs0OrU6zw1BkgnJ6sQ8CAbp2XJ7spYIgQFoRaEjAqCAXLlp3Rg",
    https=True
)

collections = {
    "sparse": "sparse_vectors",
    "zero": "zero_vectors",
    "uniform": "uniform_vectors",
    "normal": "normal_vectors"
}

def create_collection_if_not_exists(collection_name, vector_size):
    if not client.collection_exists(collection_name):
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            # vectors_config={
            #     "default": VectorParams(size=vector_size, distance=Distance.COSINE)
            # }
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
    else:
        print(f"Collection '{collection_name}' already exists.")
        
# For sparse and zero vectors, the vector dimension equals n_records; for uniform and normal, dimension is 10.
create_collection_if_not_exists(collections["sparse"], n_records)
create_collection_if_not_exists(collections["zero"], n_records)
create_collection_if_not_exists(collections["uniform"], 10)
create_collection_if_not_exists(collections["normal"], 10)



# ---------------------------
# Upsert Data into Each Collection
# ---------------------------
points_sparse, points_zero, points_uniform, points_normal = [], [], [], []

for _, row in df.iterrows():
    base_payload = {
        "Name": row["Name"],
        "Age": row["Age"],
    }

    points_sparse.append(PointStruct(id=int(row["ID"]), vector=row["sparse_vector"], payload=base_payload))
    points_zero.append(PointStruct(id=int(row["ID"]), vector=row["zero_vector"], payload=base_payload))
    points_uniform.append(PointStruct(id=int(row["ID"]), vector=row["uniform_vector"], payload=base_payload))
    points_normal.append(PointStruct(id=int(row["ID"]), vector=row["normal_vector"], payload=base_payload))

# Upsert to Qdrant collections
client.upsert(collection_name=collections["sparse"], points=points_sparse)
client.upsert(collection_name=collections["zero"], points=points_zero)
client.upsert(collection_name=collections["uniform"], points=points_uniform)
client.upsert(collection_name=collections["normal"], points=points_normal)

print("Data upserted into Qdrant Cloud collections successfully!")
