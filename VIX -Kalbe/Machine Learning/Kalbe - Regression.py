#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# ### Data Customer

# In[1]:


import pandas as pd
import numpy as np

# load dataset 
path = 'Case Study - Customer.csv'
customer = pd.read_csv(path, index_col=0, sep=";").reset_index()
customer.head(5)


# In[2]:


customer.info()


# In[3]:


customer.isnull().sum()


# In[4]:


mode = customer["Marital Status"].mode()[0]
customer["Marital Status"].fillna(mode, inplace=True)


# In[5]:


customer.isnull().sum()


# In[6]:


customer.duplicated().sum()


# In[7]:


customer.drop_duplicates(inplace=True)


# In[8]:


customer["Income"] = customer["Income"].str.replace(",", ".").astype(float)


# In[9]:


customer.head(5)


# ## Data Product

# In[10]:


# load dataset 
path = 'Case Study - Product.csv'
product = pd.read_csv(path, index_col=0, sep=";").reset_index()
product.head(10)


# In[11]:


product.info()


# ## Data Store

# In[12]:


# load dataset 
path = 'Case Study - Store.csv'
store = pd.read_csv(path, index_col=0, sep=";").reset_index()
store.head(10)


# In[13]:


store.info()


# In[14]:


store["Latitude"] = store["Latitude"].str.replace(",", ".").astype(float)
store["Longitude"] = store["Longitude"].str.replace(",", ".").astype(float)

store.head(5)


# ## Data Transaksi

# In[15]:


# load dataset 
path = 'Case Study - Transaction.csv'
transaction = pd.read_csv(path, index_col=0, sep=";").reset_index()
transaction.head(5)


# In[16]:


transaction.info()


# In[17]:


transaction.describe()


# In[18]:


transaction.isnull().sum()


# In[19]:


transaction.duplicated().sum()


# In[20]:


transaction['Date'] = pd.to_datetime(transaction['Date'])


# ## Penggabungan data menggunakan merge

# In[21]:


transaction.head(10)


# In[22]:


transaction_store = transaction.merge(store, on='StoreID', how='inner')
transaction_store_customer = transaction_store.merge(customer, on='CustomerID', how='inner')
df = transaction_store_customer.merge(product, on='ProductID', how='inner')

df.head(10)


# In[23]:


df.info()


# In[24]:


categorical = df.select_dtypes(exclude=[np.number])
categorical.columns


# In[25]:


numerical = df.select_dtypes(include=[np.number])
numerical.columns


# In[26]:


df['Date'] = pd.to_datetime(df['Date'])
df.head(5)


# In[ ]:





# In[27]:


df = df.groupby('Date')['Qty'].sum().reset_index()


# In[28]:


df.info()


# In[29]:


df.head(10)


# # Regression using ARIMA

# In[30]:


import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot

import warnings
warnings.filterwarnings("ignore")


# In[31]:


figure = px.line(df, y='Qty', x="Date", title='Jumlah Barang Terjual Dalam 1 Tahun')
figure.show()


# ## stationary test

# In[32]:


df['Date']=pd.to_datetime(df['Date'],infer_datetime_format=True)
index=df.set_index(['Date'])
from datetime import datetime
index.head()


# In[33]:


from statsmodels.tsa.stattools import adfuller
print("Observations of Dickey-fuller test")
dftest = adfuller(df['Qty'],autolag='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)


# In[34]:


decomposed = seasonal_decompose(df.set_index('Date'))
plt.figure(figsize=(20,15))

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')

plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonal')

plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Resid')


# In[35]:


# Split the data into train and test sets
train, test = df[:293], df[293:]


# In[36]:


# Create a line chart for the entire data
figure = px.line(df, x='Date', y='Qty', title='Jumlah Barang Terjual Dalam 1 Tahun')

# Add the train data to the existing figure
figure.add_scatter(x=train['Date'], y=train['Qty'], mode='lines', name='Train')

# Add the test data to the existing figure
figure.add_scatter(x=test['Date'], y=test['Qty'], mode='lines', name='Test')

# Calculate the trend line using numpy.polyfit()
trend_coefficients = np.polyfit(df.index, df['Qty'], 1)
trend_line = trend_coefficients[0] * df.index + trend_coefficients[1]

# Add the trend line to the figure
figure.add_scatter(x=df['Date'], y=trend_line, mode='lines', name='Trend Line', line=dict(color='blue', dash='dash'))

# Show the figure
figure.show()


# In[37]:


autocorrelation_plot(df['Qty'])


# In[38]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Ambil data 'Qty' dari DataFrame
data = df['Qty']

# Plot ACF dan PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot ACF
plot_acf(data, lags=40, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF)')

# Plot PACF
plot_pacf(data, lags=40, ax=ax2)
ax2.set_title('Partial Autocorrelation Function (PACF)')

plt.show()


# ## ARIMA Base Line Model

# In[39]:


train = train.set_index('Date')
test = test.set_index('Date')

y = df['Qty']

model = ARIMA(y, order=(42,0,2))
model = model.fit()

y_pred = model.get_forecast(len(test))

y_pred_df = y_pred.conf_int()
y_pred_df['prediction'] = model.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df['prediction']


# In[40]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

y_actual = test['Qty']

# Calculating the metrics
mae = mean_absolute_error(y_actual, y_pred_out)
mse = mean_squared_error(y_actual, y_pred_out)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_actual - y_pred_out) / y_actual)) * 100

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# In[41]:


# Visual inspection
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(test.index, y_actual, label='Actual')
plt.plot(test.index, y_pred_out, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Qty')
plt.legend()
plt.title('Actual vs. Predicted')
plt.show()


# In[42]:


plt.figure(figsize=(10,6))
plt.plot(train['Qty'], label='Data Training')
plt.plot(test['Qty'], color='green', label='Data Testing')
plt.plot(y_pred_out, color='black', label='ARIMA Prediction Result')
plt.title('Data Penjualan dalam 1 Tahun')
plt.legend()


# ## HYPERPARAMETER TUNING

# In[43]:




# import itertools

# # Tentukan rentang nilai untuk setiap parameter p, d, dan q
# p_values = range(0, 50)  # Contoh: 0, 1, 2, 3, 4, 5
# d_values = range(0, 1)  # Contoh: 0, 1, 2
# q_values = range(0, 10)  # Contoh: 0, 1, 2, 3, 4, 5

# best_rmse = float('inf')
# best_params = None

# # Iterasi melalui seluruh kombinasi nilai p, d, dan q dari rentang yang ditentukan
# for p, d, q in itertools.product(p_values, d_values, q_values):
#     try:
#         # Membuat model ARIMA dengan parameter (p, d, q)
#         model = ARIMA(y, order=(p, d, q))

#         # Melatih model ARIMA
#         model_fit = model.fit()

#         # Membuat prediksi untuk data latih
#         y_pred_train = model_fit.predict()

#         # Menghitung RMSE untuk prediksi pada data latih
#         rmse = mean_squared_error(y, y_pred_train, squared=False)

#         # Periksa apakah parameter saat ini memberikan RMSE terbaik
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_params = (p, d, q)
            
#     except:
#         continue

# # Menampilkan parameter terbaik dan nilai RMSE terbaik
# print(f"Best Parameters: (p={best_params[0]}, d={best_params[1]}, q={best_params[2]})")
# print(f"Best RMSE: {best_rmse:.2f}")


# # Arima

# In[44]:


y = df['Qty']

model = ARIMA(y, order=(49, 0, 6))
model = model.fit()

y_pred = model.get_forecast(len(test))

y_pred_df = y_pred.conf_int()
y_pred_df['prediction'] = model.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df['prediction']


# In[45]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

y_actual = test['Qty']

# Calculating the metrics
mae = mean_absolute_error(y_actual, y_pred_out)
mse = mean_squared_error(y_actual, y_pred_out)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_actual - y_pred_out) / y_actual)) * 100

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# In[46]:


# Visual inspection
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(test.index, y_actual, label='Actual')
plt.plot(test.index, y_pred_out, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Qty')
plt.legend()
plt.title('Actual vs. Predicted')
plt.show()


# In[47]:


plt.figure(figsize=(10,6))
plt.plot(train['Qty'], label='Data Training')
plt.plot(test['Qty'], color='green', label='Data Testing')
plt.plot(y_pred_out, color='black', label='ARIMA Prediction Result')
plt.title('Data Penjualan dalam 1 Tahun')
plt.legend()


# ## SARIMAX Model

# In[48]:


import statsmodels.api as sm
model = sm.tsa.statespace.SARIMAX(df['Qty'], order=(49, 0, 6), seasonal_order=(0, 0, 0, 7))
results = model.fit()


# In[49]:


y_pred = results.get_forecast(len(test))


# In[50]:


y_pred_df = y_pred.conf_int()
y_pred_df['forecast'] = results.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df['forecast']


# In[51]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

y_actual = test['Qty']

# Calculating the metrics
mae = mean_absolute_error(y_actual, y_pred_out)
mse = mean_squared_error(y_actual, y_pred_out)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_actual - y_pred_out) / y_actual)) * 100

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# In[52]:


# Visual inspection
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(test.index, y_actual, label='Actual')
plt.plot(test.index, y_pred_out, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Qty')
plt.legend()
plt.title('Actual vs. Predicted')
plt.show()


# In[53]:


df['forecast']=results.predict(start=293,end=365,dynamic=True)
df[['Qty','forecast']].plot(figsize=(12,8))


# In[54]:


# model = ARIMA(y, order=(49, 0, 6))
# model = model.fit()

# y_pred = model.get_forecast(len(test))

# y_pred_df = y_pred.conf_int()
# y_pred_df['prediction'] = model.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
# y_pred_df.index = test.index
# y_pred_out = y_pred_df['prediction']


# In[55]:


# # Visual inspection
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.plot(test.index, y_actual, label='Actual')
# plt.plot(test.index, y_pred_out, label='Predicted', linestyle='--')
# plt.xlabel('Date')
# plt.ylabel('Qty')
# plt.legend()
# plt.title('Actual vs. Predicted')
# plt.show()


# In[ ]:




