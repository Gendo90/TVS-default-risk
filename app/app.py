# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from equi_loans import equi_loans_val
import xgboost

#setup predictions from model
scaler_filename = 'full_data_scaler.save'
X_scaler = joblib.load(scaler_filename)

app = Flask(__name__)

# Load the model
filename = './model/XGBoost_custom_scorer.sav'
model = joblib.load(filename)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    # Get the data from the POST request.
    if request.method == "POST":
        #data = request.get_json(force=True)
        checker = [a for a in request.form.keys()]
        print(checker)
        data_cols = ['EMI', 'loan_amount', 'maximum_amount_sanctioned', 'age', 'rate_of_interest',
                     'past_due_30', 'number_of_loans', 'maximum_sanctioned', 'past_due_90', 'tenure', 
                     'times_bounced']
        data = [request.form[a] for a in data_cols]
        data = X_scaler.transform(data)
        
        # Make prediction using model loaded from disk as per the data.
        prediction = model.predict([data])

        print("Data", prediction)

        # Take the first value of prediction
        # output = prediction[0]

        return render_template("results.html", output=checker)#, exp=data)

if __name__ == '__main__':
    app.run(debug=True)
