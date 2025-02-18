import json
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from settings import settings  # Import class
from groq import Groq  # Using Groq LLM (Llama3.2)



# Initialize Qdrant Client
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
collection_name = "names1_withsyn_withoutweights_zerovectors"

# Load stored vectors from Qdrant
stored_points = client.scroll(collection_name=collection_name,with_vectors=True, limit=1000)[0]

if not stored_points:
    print("No data found in Qdrant! Check if your vector database is populated.")
    exit()

print(f" Loaded {len(stored_points)} points from Qdrant.")

# Initialize LLM Client (Groq)
groq_client = Groq(api_key=settings.GROQ_KEY)  # Ensure you have the correct API key

# ----------------------
# Function to Convert Query to Vector
# ----------------------
def convert_query_to_vector(query_name, points):
    """Convert user query into a vector using an embedding model."""
    
    # Ensure we have stored points
    if not points:
        print(" No points found in Qdrant!")
        return None

    if not points[0].vector:
        print(" Error: Vectors are missing from the retrieved points in Qdrant!")
        return None

    unique_names = [point.payload["name"] for point in points]
    query_vector = [0.0] * len(points[0].vector)  # Ensure correct dimension
    
    for idx, other_name in enumerate(unique_names):
        if query_name.lower() == other_name.lower():
            query_vector[idx] = 1.0  # Full match
        else:
            for point in points:
                synonyms = point.payload.get("synonyms", {})
                if query_name.lower() in synonyms:
                    query_vector[idx] = synonyms[query_name.lower()]
                    break
            else:
                query_vector[idx] = 0.1  # Low similarity if no match

    if not any(query_vector):
        print(f" Could not generate a vector for '{query_name}'. Available names: {unique_names}")
        return None

    # Ensure the vector matches stored vector dimensions
    if len(query_vector) != len(points[0].vector):
        print(f" Query vector size mismatch! Expected {len(points[0].vector)}, got {len(query_vector)}")
        return None

    print(f" Generated vector for '{query_name}': {query_vector}")
    return query_vector


# ----------------------
# SEARCH FUNCTION
# ----------------------
def search_name_similarity(query_name, top_k=10):
    """Convert query to vector & search Qdrant for similar names."""
    
    query_vector = convert_query_to_vector(query_name, stored_points)

    if query_vector is None:
        return None

    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        with_vectors = True,
        limit=top_k
    )

    if not search_results:
        print(f" No matches found for '{query_name}' in Qdrant.")
        return None

    return search_results

# ----------------------
# FUNCTION TO CALL LLM
# ----------------------
# ----------------------
# FUNCTION TO CALL LLM
# ----------------------
def query_llm(user_query, search_results):
    """Send user query + retrieved context to LLM and return a clean response."""
    
    context_data = []
    for result in search_results:
        name = result.payload['name']
        age = result.payload.get('age', 'Unknown')
        # synonyms = result.payload.get("synonyms", [])

        # context_data.append(f"Name: {name}, Age: {age}, Synonyms: {synonyms}")
        context_data.append(f"Name: {name}, Age: {age}")

    context_text = "\n".join(context_data)

    # Modify the prompt to avoid showing LLM thinking process
    prompt = f"""
    You are a helpful AI assistant. Based on the following retrieved information, answer the user query as clearly and directly as possible. Do not include any intermediate thinking steps or explanations. 

    Context:
    {context_text}

    User Query:
    {user_query}

    Provide a direct answer to the user querye.
    """

    try:
        response = groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,  # Use model from settings
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        
        # Extract and return only the final, user-friendly response
        final_answer = response.choices[0].message.content.strip()
        return final_answer

    except Exception as e:
        print(f" Error calling LLM: {str(e)}")
        return "Sorry, I couldn't process your request due to an error."


# ----------------------
# INTERACTIVE RAG CHAT
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
