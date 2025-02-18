import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from qdrant_client import QdrantClient
from groq import Groq




# -------------------------------
# Load Configuration
# -------------------------------
QDRANT_URL="https://64c9b924-a747-4c21-874f-f6901dc0431e.us-east-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY="vgywMs0OrU6zw1BkgnJ6sQ8CAbp2XJ7spYIgQFoRaEjAqCAXLlp3Rg"
GROQ_API_KEYS ="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" # Using multiple API keys
GROQ_MODEL = "llama-3.3-70b-versatile"



groq_client = Groq(api_key=GROQ_API_KEYS[0])
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, https=True)

# -------------------------------
# Collection Details
# -------------------------------
collections = {
    "1.names_withsynonyms_withweights_sparsematrices": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/1.names_withsynonyms_withweights_sparsematrices.xlsx",
    "2.names_withsynonyms_withoutweights_sparsematrices": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/2.names_withsynonyms_withoutweights_sparsematrices.xlsx",
    "3.names_withoutsynonyms_withoutweights_sparsematrices": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/3.names_withoutsynonyms_withoutweights_sparsematrices.xlsx",
    "4.names_withsynonyms_withweights_zerovectors": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/4.names_withsynonyms_withweights_zerovectors.xlsx",
    "5.names_withsynonyms_withoutweights_zerovectors": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/5.names_withsynonyms_withoutweights_zerovectors.xlsx",
    "6.names_withoutsynonyms_withoutweights_zerovectors": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/6.names_withoutsynonyms_withoutweights_zerovectors.xlsx"
}

# -------------------------------
# Streamlit UI Layout
# -------------------------------
st.set_page_config(layout="wide")  # Use a wide layout
st.title("Custom Embeddings Chat Interface")

# Sidebar for Collection Selection & Data Preview
with st.sidebar:
    st.header("Collection Selection")
    selected_collection = st.selectbox("Select a collection:", options=collections)

    st.subheader("Excel File Preview")
    def load_excel_from_github(url):
        """Download and load an Excel file from GitHub."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return pd.read_excel(BytesIO(response.content), engine="openpyxl")
        except Exception as e:
            return str(e)

    if selected_collection:
        excel_data = load_excel_from_github(collections[selected_collection])
        if isinstance(excel_data, str):
            st.error(f"Error loading Excel file: {excel_data}")
        else:
            st.dataframe(excel_data)

# -------------------------------
# Query Input Section
# -------------------------------
query = st.text_input("Enter your query:")

# -------------------------------
# Helper Functions
# -------------------------------
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

def search_qdrant(query_vector, collection_name, top_k=10):
    """Search in Qdrant using the generated query vector."""
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
    """Generate response using the LLM model."""
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
    """Retrieve data from the Qdrant collection."""
    stored_points = client.scroll(collection_name=collection_name, with_vectors=True, limit=1000)[0]
    if not stored_points:
        return None
    return stored_points

# -------------------------------
# Streamlit Submit Button
# -------------------------------
if st.button("Submit Query"):
    if query:
        context = ""  
        collection_data = get_collection_data(selected_collection)
        
        if collection_data:
            query_vector = convert_query_to_vector(query, collection_data)
            search_results = search_qdrant(query_vector, selected_collection)
            
            # Display search results in the left sidebar
            with st.sidebar:
                st.subheader("Retrieved Data")
                if search_results:
                    for result in search_results:
                        name = result.payload.get("name", "Unknown")
                        age = result.payload.get("age", "Unknown")
                        st.write(f"**Name:** {name}")
                        st.write(f"**Age:** {age}")
                        st.write("---")
                        context += f"{name} (Age: {age}) | "

                else:
                    st.warning("No matching results found.")

            # Call Groq LLM
            llm_response = get_llm_response(query, context)
            
            # Display LLM response
            st.subheader("Chat Response")
            st.success(llm_response)

        else:
            st.warning(f"No data found in collection {selected_collection}.")
    else:
        st.warning("Please enter a query.")
