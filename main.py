import pickle
from urllib import request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#scaler = pickle.load(open('scaler_model.pkl','rb'))
model = pickle.load(open('my_model.pkl','rb'))
df2 = pd.read_csv("Mod_acc_data.csv")

X = df2[['Category_Code', 'Accident_Type_Code', 'Year', 'Month']]
y = df2['Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='./templates')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Parse the JSON request body
#     data = request.get_json()

#     # Extract the values for each feature
#     category_code = data['category_code']
#     acc_type_code = data['accident_type_code']
#     year = data['year']
#     month = data['month']

#     # Transform the input data
#     input_data = np.array([[category_code, acc_type_code, year, month]])
#     scaled_data = scaler.transform(input_data)

#     # Make a prediction
#     prediction = model.predict(scaled_data)

#     # Return the prediction as a JSON object
#     response = {'predictions': prediction.tolist()}
#     return jsonify(response)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():

    category_code = int(request.form.get('category_code'))
    acc_type_code = int(request.form.get('acc_type_code'))
    year = int(request.form.get('year'))
    month = int(request.form.get('month'))

    #prediction
    input_data = np.array([[category_code, acc_type_code, year, month]])
    scaled_data = scaler.transform(input_data)
    result = model.predict(scaled_data)
    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

