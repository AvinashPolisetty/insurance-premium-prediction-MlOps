from flask import Flask,request,jsonify,render_template
from src.pipeline.prediction_pipeline import PredictionPipeline,CustomData

from src.logger import logging
from src.exception import CustomException
import sys


app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        data = CustomData(
            age=float(request.form.get('age')),
            sex=request.form.get('sex'),
            bmi=float(request.form.get('bmi')),
            children=float(request.form.get('children')),
            smoker=request.form.get('smoker'),
            region=request.form.get('region'),
            
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template('results.html', final_result=results)


if __name__ == "__main__":
    app.run(host='0.0.0.0')

