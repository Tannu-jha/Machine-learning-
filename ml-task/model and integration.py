#create a model and integrate with web app
from flask import Flask, request, jsonify
from flask_cors import CORS 
def predict_match(features):
    try:
        model = joblib.load('cricket_model.pkl')
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return f"Prediction: {int(prediction[0])}"
    except Exception as e:
        return f"Error: {e}"

model = joblib.load('cricket_model.pkl')
app = Flask(__name__)
CORS(app)  

@app.route('/')
def home():
    return "Welcome to the Cricket Match Predictor!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request data'}), 400
        
        features = data['features']
        
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        
        # Return the prediction result
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0")
    