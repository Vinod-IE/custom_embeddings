import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from qdrant_client import QdrantClient
from groq import Groq
from Levenshtein import ratio
from sentence_transformers import SentenceTransformer

# -------------------------------
# Set Page Config (must be the first Streamlit command)
# -------------------------------
st.set_page_config(layout="wide")

# -------------------------------
# Load Configuration
# -------------------------------
QDRANT_URL = "https://64c9b924-a747-4c21-874f-f6901dc0431e.us-east-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "vgywMs0OrU6zw1BkgnJ6sQ8CAbp2XJ7spYIgQFoRaEjAqCAXLlp3Rg"
GROQ_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Initialize clients
groq_client = Groq(api_key=GROQ_KEY)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, https=True)

# Load Sentence Transformer model (for collections using SentenceTransformers)
model_st = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# Collection Details
# -------------------------------

collections = {
    "1.names_withsynonyms_withweights_sparsematrices": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/1.names_withsynonyms_withweights_sparsematrices.xlsx",
    "2.names_withsynonyms_withoutweights_sparsematrices": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/2.names_withsynonyms_withoutweights_sparsematrices.xlsx",
    "3.names_withoutsynonyms_withoutweights_sparsematrices": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/3.names_withoutsynonyms_withoutweights_sparsematrices.xlsx",
    "4.names_withsynonyms_withweights_zerovectors": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/4.names_withsynonyms_withweights_zerovectors.xlsx",
    "5.names_withsynonyms_withoutweights_zerovectors": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/5.names_withsynonyms_withoutweights_zerovectors.xlsx",
    "6.names_withoutsynonyms_withoutweights_zerovectors": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/6.names_withoutsynonyms_withoutweights_zerovectors.xlsx",
    "7.names_withsynonyms_withweights_withbasevectors": "https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/7.names_withsynonyms_withweights_withbasevectors.xlsx",
    "8.names_withsynonyms_withoutweights_withbasevectors":"https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/8.names_withsynonyms_withoutweights_withbasevectors",
    "9.names_withoutsynonyms_withoutweights_withbasevectors":"https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/9.names_withoutsynonyms_withoutweights_withbasevectors.xlsx",
    "10.names_withsynonyms_withweights_withoutvecotrs_withSentenceTransformers":"https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/10.names_withsynonyms_withweights_withoutvecotrs_withSentenceTransformers.xlsx",
    "11.names_withsynonyms_withoutweights_withoutvectors_withSentenceTransformers":"https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/11.names_withsynonyms_withoutweights_withoutvectors_withSentenceTransformers.xlsx",
    "12.names_withoutsynonyms_withoutvectors_withSentenceTransformers":"https://raw.githubusercontent.com/Vinod-IE/custom_embeddings/main/data/12.names_withoutsynonyms_withoutvectors_withSentenceTransformers.xlsx"
      
 }

# collections = {
#     "1.names_withsynonyms_withweights_sparsematrices": os.path.join("data", "1.names_withsynonyms_withweights_sparsematrices.xlsx"),
#     "2.names_withsynonyms_withoutweights_sparsematrices": os.path.join("data", "2.names_withsynonyms_withoutweights_sparsematrices.xlsx"),
#     "3.names_withoutsynonyms_withoutweights_sparsematrices": os.path.join("data", "3.names_withoutsynonyms_withoutweights_sparsematrices.xlsx"),
#     "4.names_withsynonyms_withweights_zerovectors": os.path.join("data", "4.names_withsynonyms_withweights_zerovectors.xlsx"),
#     "5.names_withsynonyms_withoutweights_zerovectors": os.path.join("data", "5.names_withsynonyms_withoutweights_zerovectors.xlsx"),
#     "6.names_withoutsynonyms_withoutweights_zerovectors": os.path.join("data", "6.names_withoutsynonyms_withoutweights_zerovectors.xlsx"),
#     "7.names_withsynonyms_withweights_withbasevectors": os.path.join("data", "7.names_withsynonyms_withweights_withbasevectors.xlsx"),
#     "8.names_withsynonyms_withoutweights_withbasevectors": os.path.join("data", "8.names_withsynonyms_withoutweights_withbasevectors.xlsx"),
#     "9.names_withoutsynonyms_withoutweights_withbasevectors": os.path.join("data", "9.names_withoutsynonyms_withoutweights_withbasevectors.xlsx"),
#     "10.names_withsynonyms_withweights_withoutvecotrs_withSentenceTransformers": os.path.join("data", "10.names_withsynonyms_withweights_withoutvecotrs_withSentenceTransformers.xlsx"),
#     "11.names_withsynonyms_withoutweights_withoutvectors_withSentenceTransformers": os.path.join("data", "11.names_withsynonyms_withoutweights_withoutvectors_withSentenceTransformers.xlsx"),
#     "12.names_withoutsynonyms_withoutvectors_withSentenceTransformers": os.path.join("data", "12.names_withoutsynonyms_withoutvectors_withSentenceTransformers.xlsx")
# }

# -------------------------------
# Initialize Session State for Chat History
# -------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------------------
# Custom CSS (for sticky tabs and scrollable areas)
# -------------------------------
st.markdown("""
    <style>
    [data-baseweb="tablist"] {
        position: sticky;
        top: 0;
        z-index: 1000;
        background-color: white;
    }
    .scrollable-container {
        height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Streamlit UI Layout
# -------------------------------
st.title("Custom Embeddings Chat Interface")

# Sidebar: Collection Selection
with st.sidebar:
    st.header("Collection Selection")
    selected_collection = st.selectbox("Select a collection:", options=list(collections.keys()))

def load_excel(file_path):
    """Load an Excel file from a local path."""
    try:
        return pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        return str(e)

# Create two tabs: one for Excel Preview and one for Chat Interface
tabs = st.tabs(["Excel Preview", "Chat Interface"])

with tabs[0]:
    st.header("Excel File Preview")
    if selected_collection:
        file_path = collections[selected_collection]
        if file_path.endswith(".xlsx"):
            excel_data = load_excel(file_path)
            if isinstance(excel_data, str):
                st.error(f"Error loading Excel file: {excel_data}")
            else:
                st.dataframe(excel_data, use_container_width=True)
        else:
            st.info("Excel preview not available for this collection.")

with tabs[1]:
    st.header("Query and Chat")
    query = st.text_input("Enter your query:")
    
    # Temperature slider for the LLM response (lower values for more direct answers)
    # temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
    #                         help="Lower temperature produces more direct and correct details.")

    # -------------------------------
    # Helper Functions
    # -------------------------------
    def convert_query_to_vector(query_name, points):
        """
        Convert query into a vector using Levenshtein similarity (for collections 1-9).
        (For these collections the stored vectors are random 10-dim values.)
        """
        if not points:
            return None

        query_vector = []
        query_name_lower = query_name.lower()

        for point in points:
            other_name = point.payload["name"].lower()
            synonyms = point.payload.get("synonyms", [])

            # Exact or synonym match
            if query_name_lower == other_name or query_name_lower in [syn.lower() for syn in synonyms]:
                similarity_score = 1.0  
            else:
                # Compute similarity using Levenshtein Distance
                similarity_score = ratio(query_name_lower, other_name)
                print(f"Levenshtein('{query_name_lower}', '{other_name}') = {similarity_score}")

                # Boost for substring match
                if query_name_lower in other_name or other_name in query_name_lower:
                    similarity_score = max(similarity_score, 0.9)
                    print(f"Boosted for substring match: {similarity_score}")

                # Token overlap boost
                query_tokens = set(query_name_lower.split())
                other_tokens = set(other_name.split())
                token_overlap = len(query_tokens & other_tokens) / max(len(query_tokens), len(other_tokens), 1)
                print(f"Token Overlap('{query_name_lower}', '{other_name}') = {token_overlap}")

                similarity_score = max(similarity_score, token_overlap)
                print(f"Final Score: {similarity_score}\n")

            query_vector.append(similarity_score)
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
        """
        Generate response using the LLM model with instructions to provide correct and direct details.
        The temperature parameter controls the randomness.
        """
        prompt = f"User Query: {query}\nContext: {contexts}\nPlease provide a correct, direct, and detailed answer. Answer:"
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
    # Query Processing
    # -------------------------------
    if st.button("Submit Query"):
        if query:
            context = ""
            collection_data = get_collection_data(selected_collection)
            
            if collection_data:
                # For collections 10, 11, and 12, use SentenceTransformer embeddings.
                if (selected_collection.startswith("10.") or 
                    selected_collection.startswith("11.") or 
                    selected_collection.startswith("12.")):
                    query_vector = model_st.encode(query).tolist()
                else:
                    query_vector = convert_query_to_vector(query, collection_data)
                
                search_results = search_qdrant(query_vector, selected_collection)
                
                # Display search results in the sidebar
                with st.sidebar:
                    st.subheader("Retrieved Data")
                    if search_results:
                        for idx, result in enumerate(search_results, start=1):
                            name = result.payload.get("name", "Unknown")
                            age = result.payload.get("age", "Unknown")
                            score = result.score  # Retrieve similarity score

                            st.write(f"**Rank {idx}:** {name}")
                            st.write(f"**Age:** {age}")
                            st.write(f"**Similarity Score:** {score:.4f}")
                            st.write("---")

                            context += f"{name} (Age: {age}) | "
                            print(f"Rank {idx}: Name={name}, Age={age}, Score={score:.4f}")
                    else:
                        st.warning("No matching results found.")

                # Get response from the LLM model
                llm_response = get_llm_response(query, context, temperature)
                st.subheader("Chat Response")
                st.success(llm_response)

                # Save the Q&A pair in the session history (newest at the top)
                st.session_state.history.insert(0, {"question": query, "answer": llm_response})
            else:
                st.warning(f"No data found in collection {selected_collection}.")
        else:
            st.warning("Please enter a query.")

    # -------------------------------
    # Display Conversation History (with scrolling)
    # -------------------------------
    st.subheader("Conversation History")
    if st.session_state.history:
        for entry in st.session_state.history:
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(f"**A:** {entry['answer']}")
            st.write("---")
    else:
        st.write("No conversation history yet.")
