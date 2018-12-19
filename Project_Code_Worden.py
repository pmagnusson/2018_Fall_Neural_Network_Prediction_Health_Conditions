#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
# In[2]:


#Setting up model conditions for learning 

learning_rate = 0.01
epochs_value = 500
batch_size_value = 128


# In[80]:


# Importing the dataset

dataset = pd.read_csv('K:/ProjectData.csv',sep = '|', low_memory='FALSE')

dataset['All Zeros'] = dataset.loc[:,'HCC1':'HCC138'].sum(axis=1)
dataset = dataset.drop(dataset[dataset['All Zeros']==0].index)
# Check for any columns where all values are 0
dataset.loc[:,(dataset == 0).all(axis=0)]
# Remove HCC138 column and 'All Zeros' columns
dataset = dataset.drop(['HCC138','All Zeros'], axis=1)

# In[81]:


#Chopping up data into a 1 year slice

dataset = dataset.loc[dataset['MMESource'] == 2016.0]
dataset.head(10)


# In[82]:


# Creating matrix of features (X) 
# Creating matrix of target variable (Y)

X = dataset.drop(['MMESource','BISource','ID','HCC75'], 1)
#Converting all data to float
dataset[['Age',	'Gender',	'IS_HOSPICE_FLAG',	'IS_SNP_FLAG',	'HCC1',	'HCC2',	'HCC6',	'HCC8',	'HCC9',	'HCC10',	'HCC11',	'HCC12',	'HCC17',	'HCC18',	
         'HCC19',	'HCC21',	'HCC22',	'HCC23',	'HCC27',	'HCC28',	'HCC29',	'HCC33',	'HCC34',	'HCC35',	'HCC39',	'HCC40',	'HCC46',	
         'HCC47',	'HCC48',	'HCC54',	'HCC55',	'HCC57',	'HCC58',	'HCC70',	'HCC71',	'HCC72',	'HCC73',	'HCC74',	'HCC75',	'HCC76',	
         'HCC77',	'HCC78',	'HCC79',	'HCC80',	'HCC82',	'HCC83',	'HCC84',	'HCC85',	'HCC86',	'HCC87',	'HCC88',	'HCC96',	'HCC99',	
         'HCC100',	'HCC103',	'HCC104',	'HCC106',	'HCC107',	'HCC108',	'HCC110',	'HCC111',	'HCC112',	'HCC114',	'HCC115',	'HCC122',	'HCC124',	
         'HCC134',	'HCC135',	'HCC136',	'HCC137',	'HCC157',	'HCC158',	'HCC161',	'HCC162',	'HCC166',	'HCC167',	'HCC169',	'HCC170',	'HCC173',	
         'HCC176',	'HCC186',	'HCC188',	'HCC189'
]] = dataset[['Age',	'Gender',	'IS_HOSPICE_FLAG',	'IS_SNP_FLAG',	'HCC1',	'HCC2',	'HCC6',	'HCC8',	'HCC9',	'HCC10',	'HCC11',	'HCC12',	'HCC17',	'HCC18',	
         'HCC19',	'HCC21',	'HCC22',	'HCC23',	'HCC27',	'HCC28',	'HCC29',	'HCC33',	'HCC34',	'HCC35',	'HCC39',	'HCC40',	'HCC46',	
         'HCC47',	'HCC48',	'HCC54',	'HCC55',	'HCC57',	'HCC58',	'HCC70',	'HCC71',	'HCC72',	'HCC73',	'HCC74',	'HCC75',	'HCC76',	
         'HCC77',	'HCC78',	'HCC79',	'HCC80',	'HCC82',	'HCC83',	'HCC84',	'HCC85',	'HCC86',	'HCC87',	'HCC88',	'HCC96',	'HCC99',	
         'HCC100',	'HCC103',	'HCC104',	'HCC106',	'HCC107',	'HCC108',	'HCC110',	'HCC111',	'HCC112',	'HCC114',	'HCC115',	'HCC122',	'HCC124',	
         'HCC134',	'HCC135',	'HCC136',	'HCC137',	'HCC157',	'HCC158',	'HCC161',	'HCC162',	'HCC166',	'HCC167',	'HCC169',	'HCC170',	'HCC173',	
         'HCC176',	'HCC186',	'HCC188',	'HCC189']].astype("float")
y = dataset.loc[:, ['HCC75']]
dataset[['HCC75']] = dataset[['HCC75']].astype("float")


# In[83]:


# Splitting the dataset into the Training set and Test set
# Using the train_test_split from sklearn to easily split our file

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[117]:



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(600, input_dim=82, activation='relu')) 
model.add(tf.keras.layers.Dense(300, kernel_regularizer=regularizers.l1(learning_rate), activation='relu'))
model.add(tf.keras.layers.Dense(150, kernel_regularizer=regularizers.l1(learning_rate), activity_regularizer = regularizers.l2(learning_rate), activation='relu'))
model.add(tf.keras.layers.Dense(75, kernel_regularizer=regularizers.l1(learning_rate), activity_regularizer = regularizers.l2(learning_rate), activation='relu'))
model.add(tf.keras.layers.Dense(1, kernel_regularizer=regularizers.l1(learning_rate), activity_regularizer = regularizers.l2(learning_rate), activation='sigmoid'))
model.summary()


# In[118]:


# Compiling Neural Network
model.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])


# In[119]:


# Fitting our model 
HCCModel = model.fit(X_train, 
                          y_train, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          validation_data=(X_test, y_test))


# In[120]:


model.save('HCCGuess.keras', overwrite=True)  


 # In[121]:


score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[122]:


plt.plot(range(1,epochs_value+1), HCCModel.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[123]:


# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.6)


# In[124]:


# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[125]:


#Set up for PPV, not sensitivity
cm[1,1]/(cm[0,1]+cm[1,1])

