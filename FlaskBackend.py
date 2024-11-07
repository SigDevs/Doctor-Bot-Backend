from flask import Flask, request, jsonify
from flask_cors import CORS
from model import get_model_response

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    if user_input:
        response = get_model_response(user_input)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'No input provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
