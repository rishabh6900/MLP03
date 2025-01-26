import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        CO_AQI_Value: float,
        Ozone_AQI_Value: float,
        NO2_AQI_Value: float,
        PM2_5_AQI_Value: float,
        latitude: float,
        Liquefied_Natural_Gas: float
    ):
    
        self.CO_AQI_Value = CO_AQI_Value
        self.Ozone_AQI_Value = Ozone_AQI_Value
        self.NO2_AQI_Value = NO2_AQI_Value
        self.PM2_5_AQI_Value = PM2_5_AQI_Value
        self.latitude = latitude
        self.Liquefied_Natural_Gas = Liquefied_Natural_Gas

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CO_AQI_Value": [self.CO_AQI_Value],
                "Ozone_AQI_Value": [self.Ozone_AQI_Value],
                "NO2_AQI_Value": [self.NO2_AQI_Value],
                "PM2_5_AQI_Value": [self.PM2_5_AQI_Value],
                "latitude": [self.latitude],
                "Liquefied_Natural_Gas": [self.Liquefied_Natural_Gas],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)