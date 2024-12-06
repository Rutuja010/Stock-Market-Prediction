# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 00:13:14 2024

@author: shiva
"""

# import libraries
import pickle 
import streamlit  as st
import yfinance as yf 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from datetime import date
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose 
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Title
app_name = "Stock Market Forecasting App"
st.title(app_name)
st.subheader("This app is created to forecast the stock market price of the selected company.")

# add an image from online resourse 
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Take input from the user of app about start & end date 

#sidebar 
st.sidebar.header("Select the Parameter from below")

start_date=st.sidebar.date_input('Start date',date(2010,1,1))
end_date=st.sidebar.date_input('End date',date(2015,1,1))


# symbol list
ticker_list = ["AAPL","INFY","MSFT","GOOGL","TSLA","NVDA"]

# Dropdown list
ticker=st.sidebar.selectbox("Select the company",ticker_list)


# fetch data from user inputs yfinance library 
data=yf.download(ticker,start=start_date,end=end_date)

# If the index is not a DatetimeIndex, convert it
if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)

# Now continue with your existing operations
st.write(data)

# Create the 'Volume (Thousands)' column by dividing the original 'Volume' by 1,000
data['Volume'] = data['Volume'] / 1_000

# Add date as column to the dataframe
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write("Date from",start_date,"To",end_date)
# Display the filtered data
st.write(data)
# Plot the data
st.header("Data Visualization")
st.subheader("Plot of the data")
fig=px.line(data,x="Date",y=data.columns,title="Closing price of the stock",width=800,height=800)
st.plotly_chart(fig)

# Add a selectbox to select column from the data 
column=st.selectbox("Select the column to be used for forecasting",data.columns[1:])

# Subsetting the data for selecting date vs column 
Data=data[["Date",column]]
st.write("Selected Data",Data)

# ADF test check for stationarity 
st.header("Is Data Stationary?")
st.write(adfuller(data[column])[1]<0.05)

# Decompose the data 
st.header("Decompostion of data")
decomposition=seasonal_decompose(data[column],model="multiplicative",period=12) 
st.write(decomposition.plot())

# Add the slider for selecting values of (p,d,q)
p=st.slider("Select the value of p",0,5,2)
d=st.slider("Select the value of d",0,2,1)
q=st.slider("Select the value of q",0,5,2)
seasonal_order=st.number_input("Select the value of sp:-",1,24,12)

# Using Sarimax  first trained the model
model_1= sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))

# fit the model
model_1=model_1.fit()
        
# Print the summary of the model 
st.header("Model_Summary")
st.write(model_1.summary())

# Predict the future values (forecasting)
forecast_period=st.number_input("Select the number of days to forecast",1,365,10)

predictions=model_1.get_forecast(steps=forecast_period)

# Get confidence intervals for predictions
pred_ci = predictions.conf_int()

# Extract the Predicted mean values (forecasted values)
predicted_mean=predictions.predicted_mean

# Display the predicted values and the confidence intervals
st.write("Confidence Interval:-")
st.write(pred_ci)

# Change the index of predictions column 
predicted_mean.index=pd.date_range(start=end_date,periods=forecast_period,freq="D")

# Convert to Dataframe for better display
predicted_mean_df=pd.DataFrame(predicted_mean)

# add the date into index
predicted_mean_df.insert(0, "Date",predicted_mean.index,True)
predicted_mean_df.reset_index(drop=True,inplace=True)
st.write("Predicted Data",predicted_mean_df)
st.write("Actual Data",Data)
st.write("----")

# Plot the Actual & Predicted data
fig = go.Figure()

fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode="lines",name="Actual_Data",line=dict(color="blue")))
fig.add_trace(go.Scatter(x=predicted_mean_df["Date"],y=predicted_mean_df["predicted_mean"],mode="lines",name="Predicted_Data",line=dict(color="red")))

# Add the title
fig.update_layout(title="Actual vs Predicted",xaxis_title="Date",yaxis_title="Price")

# display the plot 
st.plotly_chart(fig)

# Calculate the value of RMSE & MSE 
actual_values = data[column].iloc[-forecast_period:].values  # Last n actual values
predicted_values = predictions.predicted_mean.values  # The forecasted values

# Now calculate RMSE
mse = mean_squared_error(actual_values, predicted_values)
rmse = np.sqrt(mse)

# Display RMSE in Streamlit
st.write(f'Mean Square Error(MSE),{mse:.4f}')
st.write(f'Root Mean Square Error(RMSE),{rmse:.4f}')
