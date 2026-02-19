import numpy as np
import pickle
from flask import Flask,  render_template,request

app = Flask(__name__)

# Load model (change filename if different)
try:
    model = pickle.load(open("rainfall_model.pkl", "rb"))
except:
    model = None


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    try:
        jan_feb = float(request.form['Jan-Feb'])
        mar_may = float(request.form['Mar-May'])
        jun_sep = float(request.form['Jun-Sep'])
        oct_dec = float(request.form['Oct-Dec'])

        features = np.array([[jan_feb, mar_may, jun_sep, oct_dec]])

        if model:
            prediction = model.predict(features)[0]
        else:
            # fallback logic
            total = features.sum()
            prediction = 1 if total > 200 else 0

        if prediction == 1:
            return render_template("chance.html")
        else:
            return render_template("nochance.html")

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)
