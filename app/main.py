#All Imports
from flask import Flask, render_template, request, flash, redirect
from flask_cors import CORS
import pickle
import numpy as np
import sklearn

#App Initialisation
app = Flask(__name__)
CORS(app)

#Prediction Funvtion
model = pickle.load(open('models/liver.pkl','rb'))
def predict(values, dic):
    if len(values) > 10 or len(values) < 10:
        return -1

    values = np.asarray(values)

    return model.predict(values.reshape(1, -1))[0]

#Home Route
@app.route("/")
def home():
    return render_template('home.html')

#Liver Cancer Route
@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

#Prediction Route
@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred)

@app.route("/livercancer-API", methods = ['POST', 'GET'])
def api_pred():
    pred = -1

    try:
        if request.method == 'POST':
            to_predict_dict = request.get_json()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
        else:
            to_predict_dict = request.args
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        return -1, 200

    return pred, 200

if __name__ == '__main__':
	app.run(debug = True)