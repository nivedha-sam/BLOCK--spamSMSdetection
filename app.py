from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():  
    data = request.get_json(force=True)
    message = data.get('message', '')

    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]
    result = "⚠️ Spam Detected" if prediction == 1 else "✅ Looks Normal"

    return jsonify({'prediction': result})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
