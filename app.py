from flask import Flask, render_template, request
import pickle
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

app = Flask(__name__)

# Load the trained model using pickle
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
data = pd.read_csv('diabetes.csv')  # Replace 'your_dataset.csv' with your dataset
X = data.drop('Outcome', axis=1)
features = X.columns.tolist()

# Create a LIME explainer
explainer = LimeTabularExplainer(X.values, mode="classification", feature_names=features, class_names=['0', '1'])

@app.route('/')
def index():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = []
        for feature in features:
            user_input.append(float(request.form[feature]))
        user_input =np.array([user_input]) # Format the input as a list of lists
        
        # Predict with the model
        prediction = model.predict(user_input)
        
        # Explain the prediction using LIME
        explanation = explainer.explain_instance(user_input[0], model.predict_proba, num_features=len(features))

        return render_template('index.html', prediction=prediction, explanation=explanation.as_list())

if __name__=="__main__":
    
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    #port=int(os.environ.get('PORT',5000))
    #app.run(debug=False)
    app.run(debug=True)