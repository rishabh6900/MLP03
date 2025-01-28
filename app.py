from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect data from the form
        data = CustomData(
            # AQI_Value=float(request.form.get('AQI_Value')),
            CO_AQI_Value=float(request.form.get('CO_AQI_Value')),
            Ozone_AQI_Value=float(request.form.get('Ozone_AQI_Value')),
            NO2_AQI_Value=float(request.form.get('NO2_AQI_Value')),
            PM2_5_AQI_Value=float(request.form.get('PM2_5_AQI_Value')),
            latitude=float(request.form.get('latitude')),
            Liquefied_Natural_Gas=float(request.form.get('Liquefied_Natural_Gas'))
        )

        # Convert the data into a DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Load the prediction pipeline and make predictions
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)           # yahi tumhe dhoka de rahi hai
        print("After Prediction")

        # Render results on the home page
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
