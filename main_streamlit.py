import streamlit as st
import numpy as np
from qdrant_client import QdrantClient
from groq import Groq  

# -------------------------------
# Load Configuration
# -------------------------------
QDRANT_URL = "https://64c9b924-a747-4c21-874f-f6901dc0431e.us-east-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "vgywMs0OrU6zw1BkgnJ6sQ8CAbp2XJ7spYIgQFoRaEjAqCAXLlp3Rg"
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Initialize Clients
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, https=True)
groq_client = Groq(api_key=GROQ_API_KEY)

# Collection Mapping
collections = {
    "sparse": "sparse_vectors",
    "zero": "zero_vectors",
    "uniform": "uniform_vectors",
    "normal": "normal_vectors"
}

# -------------------------------
# Streamlit Chat Interface
# -------------------------------
st.title("Qdrant & Groq LLM Chatbot")
st.write("Select an embedding method, ask your query, and get responses from Llama-3.3.")

# Dropdown to select the embedding method (collection)
selected_method = st.selectbox("Select Embedding Method", list(collections.keys()))
collection_name = collections[selected_method]

# Text input for the query
query = st.text_input("Enter your query:", "What is the age of Vinod?")

# -------------------------------
# Function to Convert Query to Vector
# -------------------------------
def query_to_vector(query, method):
    n_records = 10  # Adjust as per stored embeddings
    if method == "sparse_vectors":
        vec = [0.0] * n_records
        if "vinod" in query.lower():
            vec[1] = 1.0  # Simulating sparse vector behavior
        else:
            vec[0] = 1.0
        return vec
    elif method == "zero_vectors":
        return [0.0] * n_records
    elif method == "uniform_vectors":
        return np.random.uniform(0, 1, 10).round(2).tolist()
    elif method == "normal_vectors":
        return np.random.normal(0.5, 0.1, 10).round(2).tolist()
    else:
        return [0.0] * n_records

# -------------------------------
# Function to Generate Response using Groq LLM
# -------------------------------
def get_llm_response(query, context):
    prompt = f"User Query: {query}\nContext: {context}\nAnswer:"
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content.strip() if response.choices else "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# Handle Query Submission
# -------------------------------
if st.button("Submit Query"):
    if query:
        # Convert the query into an embedding
        query_vector = query_to_vector(query, collection_name)
        st.write("**Query Vector:**", query_vector)
        
        # Perform similarity search in Qdrant
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=1
        )
        
        # Retrieve and display the result
        if search_result:
            result = search_result[0]
            name = result.payload.get("Name", "Unknown")
            age = result.payload.get("Age", "Unknown")
            context = f"Name: {name}, Age: {age}"
            
            # Send the retrieved data to LLM
            llm_response = get_llm_response(query, context)
            st.success(f"**LLM Response:** {llm_response}")
        else:
            st.error("No matching result found.")
    else:
        st.warning("Please enter a query.")
