from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from main import search_name_similarity, query_llm
import markdown2

print("hai")

app = Flask(__name__)

CORS(app)

@app.route("/chat", methods=['POST'])
def main():
    try:
        data = request.get_json()
        query = data.get('query')
        search_results = search_name_similarity(query)
        if search_results:
            llm_response = query_llm(query, search_results)
            
            # Format the result as markdown
            formatted_answer = markdown2.markdown(llm_response, extras=["bullet"])   
            
            # Return the formatted markdown answer in JSON response
            return jsonify({
                "query": query,
                "answer": formatted_answer
            }),200

        return jsonify({
            "message":f" No results found for '{query}'. Try a different name."
            }),404
        
    except Exception as e:
        return jsonify({
            "message":f" error: {e}"
            }),500
        
        
if __name__ == "__main__":
    app.run(port=5050, debug=True, use_reloader=True)
