#!/usr/bin/env python
# coding: utf-8

# # Cryptocurrency Prediction
# 
# Although the first redistributed cryptocurrency (bitcoin) was created in 2009, the idea of digital money arised in 1980s. In the recent years, cryptocurrencies have gained tremendeous popularity. As traditional currencies, the value of cryptocurrencies are changing in time. Using the historical data, I will implement a recurrent neural netwok using LSTM (Long short-term memory) layers to predict the trend of value of a cryptocurrency in the future.
# 
# There is a huge dataset about cryptocurrency market prices on Kaggle. 
# 
# https://www.kaggle.com/jessevent/all-crypto-currencies
# 
# I will only use a part of it which is historical price data of litecoin.

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


sns.set(style="darkgrid", font_scale=1.5)


# In[2]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# After importing the libraries, we can now get the data we need:

# In[3]:


df = pd.read_csv("crypto-markets.csv", parse_dates=["date"], index_col="date")
df.shape


# In[4]:


df.head()


# As you can see, the data includes almost 1 million rows. I just need a part of which is the "open" prince of "litecoin".

# In[5]:


df = df[df.slug == "litecoin"][["open"]]
df.shape


# In[7]:


df.tail()


# We now have the historical data of litecoin from April,2013 to November,2018. Let's take a look at how it changed through time.

# In[15]:


plt.figure(figsize=(12,6))
sns.lineplot(x=df.index, y="open", data=df).set_title("Price of Litecoin")


# It was in a long time silence following a small peak through the end 2013. Then reached its highest values at the end of 2017. We can make the graph look smoother by applying down-sampling.

# In[21]:


df.resample('10D').mean().plot(figsize=(12,6))
plt.figtext(.5,0.9,"Down-sampled to 10-day periods", fontsize=20, ha='center')


# # Model

# Long short-term memory (LSTM) is a type of recurrent neural network (RNN) and powerful to model sequence data because it maintains an internal state to keep track of the data it has already seen. Common applications of LSTMs include time series analysis and natural language processing.
# 
# LSTM requires the input to be a 3D tensor with shape (batch_size, timesteps, input_dim).
# 
# Batch_size: 
# Timesteps:
# input_dim:
# 
# I will build a model that predicts the value of litecoin at time t using the past 90 values (from t-90 to t-1). Therefore the the number of timesteps is 90. I only use "open" price to make the prediction so the input_dim is 1. The target variable is "open" price of litecoin which can be effected by many other factors. For example, the value of another crptocurrency may have an effect on litecoin. If we also use a second variable to make the prediction, than the input_dim will be 2.

# ## Data Preprocessing
# 
# I will re-organize the data in a way that a sequence of the values in previous 90 days is used to predict the value at time t. It is also better to normalize the values due to excessive computations done in a neural network.

# In[75]:


data = df.iloc[:, 0]


# In[76]:


hist = []
target = []
length = 90

for i in range(len(data)-length):
    x = data[i:i+length]
    y = data[i+length]
    hist.append(x)
    target.append(y)


# In[82]:


print(len(hist[0]))
print(len(hist))
print(len(target))


# In[83]:


print(hist[0][89])
print(data[89])


# The last timestep of the first element is same as the 90th timestep of the original data which is what we planned to do.
# 
# The second element of hist should be just one timestep shifted version of first element. Therefore, the last timestep in the second element should be equal to the first item of the target variable which is the 91st timestep in the original data.
# 
# Let's double check to confirm:

# In[86]:


print(hist[1][89])
print(data[90])
print(target[0])


# In[87]:


print(type(hist), type(target))


# In[88]:


#convert list to array
hist = np.array(hist)
target = np.array(target)


# In[89]:


print(type(hist), type(target))


# In[90]:


hist.shape


# In[91]:


target.shape


# In[92]:


target = target.reshape(-1,1)
target.shape


# In[93]:


#Normalize the values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
hist_scaled = sc.fit_transform(hist)
target_scaled = sc.fit_transform(target)


# In[94]:


#Reshape the input
hist_scaled = hist_scaled.reshape((len(hist_scaled), length, 1))
print(hist_scaled.shape)


# ## Training and test sets
# 
# Our dataset includes 1951 samples (days). A sample of input consists of the value of litecoin for 90 consecutive days and target variable is the value of litecoin on the following day after the 90-day period.
# 
# I will split the dataset in a way that 1900 samples are used for training and then the model will be used to predict the trend in the following 51 days
# 

# In[95]:


X_train = hist_scaled[:1900,:,:]
X_test = hist_scaled[1900:,:,:]

y_train = target_scaled[:1900,:]
y_test = target_scaled[1900:,:]


# ## Building the neural network

# In[43]:


import tensorflow as tf

from tensorflow.keras import layers


# In[100]:


model = tf.keras.Sequential()

model.add(layers.LSTM(units=32, return_sequences=True,
                  input_shape=(90,1), dropout=0.2))

model.add(layers.LSTM(units=32, return_sequences=True,
                  dropout=0.2))

model.add(layers.LSTM(units=32, dropout=0.2))

model.add(layers.Dense(units=1))

model.summary()


# We have created a model with 3 LSTM layers and 1 dense layer. It is time to compile and train the model.

# In[101]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[102]:


history = model.fit(X_train, y_train, epochs=30, batch_size=32)


# It seems like the model is converged in 30 epochs so there is no need to do additional epochs. The loss is less than 0.002 which I think is pretty good. 
# 
# Let's visualize how loss changes as number of epochs increases.

# In[107]:


loss = history.history['loss']
epoch_count = range(1, len(loss) + 1)
plt.figure(figsize=(12,8))
plt.plot(epoch_count, loss, 'r--')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();


# It seems like the model has converged after 15 epochs. After that, the loss is bouncing up and down around 0.0018.
# 
# Now it is time to make predictions.

# In[108]:


pred = model.predict(X_test)


# In[109]:


plt.figure(figsize=(12,8))
plt.plot(y_test, color='blue', label='Real')
plt.plot(pred, color='red', label='Prediction')
plt.title('Litecoin Price Prediction')
plt.legend()
plt.show()


# Our model achieved to determine the trend pretty well. As you may have noticed, the values are much less than the original values because we normalized the values before training the model. We can inverse transform to reflect the real prices but trend will be the same.

# In[113]:


pred_transformed = sc.inverse_transform(pred)
y_test_transformed = sc.inverse_transform(y_test)


# In[114]:


plt.figure(figsize=(12,8))
plt.plot(y_test_transformed, color='blue', label='Real')
plt.plot(pred_transformed, color='red', label='Prediction')
plt.title('Litecoin Price Prediction')
plt.legend()
plt.show()


# There is always space for improvement. The fuel of a neural network is data so we can build a more robust and accurate model by collecting more data. We can also try to adjust number of nodes in a layer or add additional LSTM layers. We can also try to increase the number of timesteps which was 90 in our model. Another way to improve is to adjust parameters using GridSearchCV. 
# 
# Please keep in mind that it is not always good to increase the model accuracy because we may end up having an overfit model.

# In[ ]:




