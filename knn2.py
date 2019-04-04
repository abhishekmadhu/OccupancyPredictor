import pandas as pd
import datetime as dt
from sklearn.preprocessing import scale


def read_data(file_path):
    data = pd.read_csv(file_path,header = 0)
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["hour"] = data["date"].dt.hour
    data["minute"] = data["date"].dt.minute
    data["second"] = data["date"].dt.second
    data['weekday'] = data[['date']].apply(lambda x: dt.datetime.strftime(x['date'], '%A'), axis=1)
    return data

training_data = read_data('datatraining.txt')

print(training_data[:10])

subset_features = training_data[["Occupancy","Temperature","Humidity","Light","CO2","HumidityRatio"]]

scaled_data = training_data[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]]

scaled_data.loc[:, "Temperature"] = scale(scaled_data["Temperature"])
scaled_data.loc[:, "Humidity"] = scale(scaled_data["Humidity"])
scaled_data.loc[:, "Light"] = scale(scaled_data["Light"])
scaled_data.loc[:, "CO2"] = scale(scaled_data["CO2"])