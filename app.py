from flask import Flask, request, jsonify, send_from_directory
from gmailscrape import similarity_search, create_prompt, chatBot, collect_messages, get_embedding  # Import your existing function


# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/rag', methods=['POST'])
def rag_pipeline():
    try:
        data = request.get_json()
        #print(f"Received data: {data}")

        user_query = data.get('query')
        if not user_query:
            return jsonify({'error': 'No user query provided'}), 400

        #print(f"Performing similarity search for query: {user_query}")

        search_results = similarity_search(search_vector= "summary", query=user_query)

        #print(f"Creating prompt with search results: {search_results}")
        prompt = create_prompt(user_query, search_results)

        #print(f"Sending prompt to GPT: {prompt}")
        response = collect_messages(prompt)

        return jsonify({'gpt_response': response})

    except Exception as e:
        print(f"Error: {e}")  # Log the error to the console
        return jsonify({'error': str(e)}), 500
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)