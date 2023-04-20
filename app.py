from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open("predictor.pkl", "rb"))

@app.route('/')
def home():
    return 'hello world'

@app.route('/predict',methods=['GET'])
def predict():
    #return f'islam me'
    Gender = int(request.args.get('Gender'))
    Age = int(request.args.get('Age'))
    Height = float(request.args.get('Height'))
    Weight = float(request.args.get('Weight'))
    Duration = float(request.args.get('Duration'))
    Heart_Rate = float(request.args.get('Heart_Rate'))
    Body_Temp = float(request.args.get('Body_Temp'))
    
    

    input_data = np.array([Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]).reshape(1, -1)
    
    #
    #
    prediction = model.predict(input_data)[0]
    #prediction = round(prediction, 2)
    #return [Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]
    #return [prediction]
    return f"{prediction}"

if __name__ == '__main__':
    app.run(debug=True)