from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl",'rb'))

@app.route("/",methods=['GET'])
def Home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():

    # gender = request.form.get("gender")
    # age = request.form.get("age")

    float_features = [float(x) for x in request.form.values()]

        
    features = [np.array(float_features)]
    
    prediction = model.predict(features)


    return  render_template("index.html",prediction_text = f"{prediction}")








if __name__ == "__main__":
    app.run(debug=True)