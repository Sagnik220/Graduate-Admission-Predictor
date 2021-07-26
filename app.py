from flask import Flask,request, url_for, redirect, render_template
import numpy as np
import joblib

app = Flask(__name__)

model=joblib.load('linear_reg_model.pkl')

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    output='{0:.{1}f}'.format(prediction[0][0], 2)

    if output>str(0.5):
        return render_template('index.html',pred='Probability of getting admitted is {}%'.format(output),bhai="Safe to Apply")
    else:
        return render_template('index.html',pred='Probability of getting admitted {}%'.format(output),bhai="Not Safe to Apply")


if __name__ == '__main__':
    app.run(debug=True)
