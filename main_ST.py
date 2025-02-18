import json
from qdrant_client import QdrantClient
from settings import settings
from sentence_transformers import SentenceTransformer
from groq import Groq

# Initialize Qdrant Client
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
collection_name = "employee_with_ST"

# Initialize Sentence Transformer model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load stored vectors from Qdrant
stored_points = client.scroll(collection_name=collection_name, with_vectors=True, limit=1000)[0]

if not stored_points:
    print("No data found in Qdrant!")
    exit()

# ----------------------
# Function to Convert Query to Embedding
# ----------------------
def convert_query_to_embedding(query_name):
    return embedding_model.encode(query_name).tolist()

# ----------------------
# SEARCH FUNCTION
# ----------------------
def search_name_similarity(query_name, top_k=10):
    query_vector = convert_query_to_embedding(query_name)

    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        with_vectors=True,
        limit=top_k
    )
    return search_results

# ----------------------
# LLM QUERY FUNCTION
# ----------------------
def query_llm(user_query, search_results):
    context_data = []
    for result in search_results:
        name = result.payload['name']
        age = result.payload.get('age', 'Unknown')
        context_data.append(f"Name: {name}, Age: {age}")

    context_text = "\n".join(context_data)
    prompt = f"""
    You are a helpful AI assistant. Based on the following retrieved information, answer the user query as clearly and directly as possible:
    Context:
    {context_text}
    User Query:
    {user_query}
    Provide a direct answer.
    """

    groq_client = Groq(api_key=settings.GROQ_KEY)
    response = groq_client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# ----------------------
# INTERACTIVE SEARCH
# ----------------------
if __name__ == "__main__":
    while True:
        user_input = input("🔎 Enter a name to search (or type 'exit' to quit): ").strip()

        if user_input.lower() == "exit":
            print("\n Exiting... Thank you!\n")
            break

        search_results = search_name_similarity(user_input)
        
        if search_results:
            llm_response = query_llm(user_input, search_results)
            print(f"\n LLM Response:\n{llm_response}\n")
        else:
            print(f" No results found for '{user_input}'. Try a different name.\n")
