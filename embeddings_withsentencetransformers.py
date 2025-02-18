#---------using sentence transformers
import pandas as pd
import json
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from settings import settings
from sentence_transformers import SentenceTransformer

# Load Excel data
file_path = r"data\8.names_withoutsynonyms_withoutvectors.xlsx"
df = pd.read_excel(file_path)

# Convert column names to lowercase and remove spaces
df.columns = df.columns.str.lower().str.strip()

# Debug: Print column names
print("Columns in DataFrame:", df.columns.tolist())

# Ensure required columns exist
required_columns = {"id", "name", "age"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise KeyError(f"Missing columns in Excel: {missing_columns}")

# Initialize Qdrant Client
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
collection_name = "8.names_withoutsynonyms_withoutvectors"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # You can use any model you prefer

# Prepare Points for Qdrant
points = []
for index, row in df.iterrows():
    id = row["id"]
    name = row["name"]
    age = row["age"]

    # Generate embedding for the name
    vector = embedding_model.encode(name).tolist()

    # Append to points list
    points.append(PointStruct(id=index, vector=vector, payload={"name": name, "age": age}))

# Ensure the collection exists before inserting data
try:
    collection_info = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists.")
    vector_size = collection_info.config.params.vectors.size
except Exception:
    print(f"Creating collection '{collection_name}'...")
    vector_size = len(points[0].vector) if points else 0
    if vector_size > 0:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    else:
        print("Error: No valid vectors found. Cannot create collection.")
        exit()

# Upload Data to Qdrant
if points:
    client.upsert(collection_name=collection_name, points=points)
    print("Employee relationship data successfully stored in Qdrant!")

    # Verify stored data
    stored_points = client.scroll(collection_name=collection_name, limit=5, with_vectors=True)
    print("\nSample Data Stored in Qdrant:")
    for point in stored_points[0]:
        print(f"ID: {point.id}, Name: {point.payload.get('name', 'N/A')}, Vector: {point.vector[:5]}...")  # Print first 5 values for preview
else:
    print("No valid data to insert.")

# ------------------------------------
# SEARCH FUNCTION
# ------------------------------------
def search_name_similarity(query_name, top_k=3):
    """Search for similar names in Qdrant based on vector similarity."""
    
    # Generate embedding for the query name
    query_vector = embedding_model.encode(query_name).tolist()

    # Search in Qdrant
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_vectors=False
    )

    print(f"\n🔍 Search Results for '{query_name}':")
    for result in search_results:
        print(f"   - {result.payload['name']} (Score: {round(result.score, 4)})")
    print("\n")

# ------------------------------------
# INTERACTIVE SEARCH EXECUTION
# ------------------------------------
if __name__ == "__main__":
    while True:
        user_input = input("🔎 Enter a name to search (or type 'exit' to quit): ").strip()

        if user_input.lower() == "exit":
            print("\n Exiting... Thank you!\n")
            break

        search_name_similarity(user_input)
