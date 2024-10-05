from flask import Flask,request,jsonify,render_template
import pickle 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

scaler=pickle.load(open("Models/standardScalar.pkl",'rb'))
model=pickle.load(open("Models/modelForPred.pkl",'rb'))

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():

    if request.method=='POST':
        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)

        if predict[0]==1:
            result="You may have diabetes, Kindly visit hospital."
        else:
            result="Chill..!! You dont have diabetes."

        return render_template('home.html',results=result)
    
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0") 