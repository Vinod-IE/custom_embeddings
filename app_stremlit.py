import streamlit as st
import numpy as np
from qdrant_client import QdrantClient
from groq import Groq
from settings import settings

# -------------------------------
# Load Configuration
# -------------------------------
QDRANT_URL = settings.QDRANT_URL
QDRANT_API_KEY = settings.QDRANT_API_KEY
GROQ_API_KEYS = [settings.GROQ_KEY]  # Using multiple API keys
GROQ_MODEL = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=GROQ_API_KEYS[0])
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, https=True)

# List of available collections
collections = ["1.names_withsynonyms_withweights_sparsematrices", "2.names_withsynonyms_withoutweights_sparsematrices", 
               "3.names_withoutsynonyms_withoutweights_sparsematrices", "4.names_withsynonyms_withweights_zerovectors", 
               "5.names_withsynonyms_withoutweights_zerovectors", "6.names_withoutsynonyms_withoutweights_zerovectors"]


st.title("Custome Embeddings")
st.write("Enter a query, select a collection to search, and get responses.")
# Dropdown for selecting a collection
selected_collection = st.selectbox(
    "Select a collection to search:",
    options=collections
)
query = st.text_input("Enter your query:", "")



def convert_query_to_vector(query_name, points):
    """Convert user query into a vector using an embedding model."""
    if not points:
        return None

    unique_names = [point.payload["name"] for point in points]
    query_vector = [0.0] * len(points[0].vector)
    
    for idx, other_name in enumerate(unique_names):
        if query_name.lower() == other_name.lower():
            query_vector[idx] = 1.0
        else:
            for point in points:
                synonyms = point.payload.get("synonyms", {})
                if query_name.lower() in synonyms:
                    query_vector[idx] = synonyms[query_name.lower()]
                    break
            else:
                query_vector[idx] = 0.1  # Low similarity if no match

    return query_vector

def search_qdrant(query_vector, collection_name, top_k=3):
    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return search_results
    except Exception as e:
        return []

def get_llm_response(query, contexts):
    prompt = f"User Query: {query}\nContext: {contexts}\nAnswer:"
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content.strip() if response.choices else "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"

def get_collection_data(collection_name):
    """Load data from Qdrant collection."""
    stored_points = client.scroll(collection_name=collection_name, with_vectors=True, limit=1000)[0]
    if not stored_points:
        return None
    return stored_points

# ------------------------
# Streamlit Button Action
# ------------------------
if st.button("Submit Query"):
    if query:
        context = ""  # Empty context as we are no longer displaying search results
        collection_data = get_collection_data(selected_collection)
        
        if collection_data:
            query_vector = convert_query_to_vector(query, collection_data)
            search_results = search_qdrant(query_vector, selected_collection)
            
            # Build context
            if search_results:
                extracted_info = [f"{r.payload.get('name', 'Unknown')}: {r.payload.get('age', 'Unknown')}" for r in search_results]
                context += f"{selected_collection}: " + " | ".join(extracted_info) + "\n"
            else:
                context += f"{selected_collection}: No matching result.\n"
            
            # Call Groq LLM
            llm_response = get_llm_response(query, context)
            st.subheader("LLM Response:")
            st.success(llm_response)
        else:
            st.warning(f"No data found in collection {selected_collection}.")
    else:
        st.warning("Please enter a query.")
