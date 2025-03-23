from flask import Flask, request, render_template,redirect
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl', 'rb'))
model_svm = pickle.load(open('model_svm.pkl', 'rb'))
sc= pickle.load(open('scaler.pkl', 'rb'))

# creating flask app
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    n = int(request.form['Nitrogen'])
    p = int(request.form['Phosphorus'])
    k = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rain = float(request.form['Rainfall'])

    feature_list = [[n, p, k, temp, humidity, ph, rain]]
    # single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = sc.transform(feature_list)
    prediction = model_svm.predict(scaled_features)
    pred = prediction[0]

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if pred in crop_dict:
        crop = crop_dict[pred]
        result = "{} is the best crop to be cultivated right there.".format(crop)
    else:
        result = "Sorry are not able to recommend a proper crop for this environment"
    return render_template("index.html", result=result)

@app.route('/go-back', methods=['POST'])
def go_back():
    referrer = request.referrer
    if referrer:
        return redirect(referrer)
    return redirect("/")  # Fallback to home page

# python main
if __name__ == "__main__":
    app.run(debug=True)
