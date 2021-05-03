import requests
import numpy as np
import pandas as pd
import plotly.offline as py
from fbprophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt
from fbprophet.plot import plot_plotly
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
from fbprophet.plot import add_changepoints_to_plot
train = pd.read_csv('train.csv')
train['date'] = pd.to_datetime(train['date'])
app = Flask(__name__, template_folder = 'Templates')
@app.route('/', methods = ['GET'])
def Home():    
    return render_template('home.html')
standard_to = StandardScaler()
@app.route('/', methods = ['POST'])
def predict():
    if request.method == 'POST':
        sid = int(request.form['aa'])
        ino = int(request.form['ab'])
        df = train[train['store'] == sid]
        df = df[df['item'] == ino]
        X = df['date']
        y = df['sales']
        df_train = pd.DataFrame()
        df_train['ds'] = X
        df_train['y'] = y
        fbp = Prophet()
        fbp.fit(df_train)
        future = fbp.make_future_dataframe(periods = 365)
        forecast = fbp.predict(future)
        fig = fbp.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), fbp, forecast)
        plt.xlabel('Store ID: ' + str(sid) + ' Item No.: ' + str(ino))
        plt.show()
        return render_template('home.html')
    else:
        return render_template('home.html')
if __name__=="__main__":
    #app.run(host = '0.0.0.0', port = 8080)
    app.run(debug = True)