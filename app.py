from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('car_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    engine_size = float(request.form['engine_size'])
    fuel_consumption = float(request.form['fuel_consumption'])

    prediction = model.predict([[engine_size, fuel_consumption]])

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
