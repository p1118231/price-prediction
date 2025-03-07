#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
from prophet import Prophet
import socket
import pandas as pd
from prophet.serialize import model_from_json
import json


app = Flask(__name__)
# Load models
# Method 1: Using fin.read() consistently across all models
with open('daily_model.json', 'r') as fin:
    model_daily = model_from_json(fin.read())  

with open('weekly_model.json', 'r') as fin:
    model_weekly = model_from_json(fin.read())  # Changed from json.load()

with open('monthly_model.json', 'r') as fin:
    model_monthly = model_from_json(fin.read())  # Changed from json.load()


def find_free_port():
    """Finds a free port by binding to an ephemeral port assigned by the OS."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))  # Bind to a free port provided by the host.
    free_port = s.getsockname()[1]
    s.close()
    return free_port


@app.route('/predict', methods=['GET'])
def predict_all():
    today = pd.to_datetime('today')
    dates_daily = pd.date_range(start=today, periods=1, freq='D').to_frame(index=False, name='ds')
    dates_weekly = pd.date_range(start=today, periods=7, freq='D').to_frame(index=False, name='ds')
    dates_monthly = pd.date_range(start=today, periods=30, freq='D').to_frame(index=False, name='ds')

    forecast_daily = model_daily.predict(dates_daily)
    forecast_weekly = model_weekly.predict(dates_weekly)
    forecast_monthly = model_monthly.predict(dates_monthly)

    return jsonify({
        'daily_sales': int(forecast_daily['yhat'].sum()),
        'weekly_sales': int(forecast_weekly['yhat'].sum()),
        'monthly_sales': int(forecast_monthly['yhat'].sum())
    })

if __name__ == '__main__':
    port = find_free_port()
    app.run(port= port,debug=False)


# In[ ]:




