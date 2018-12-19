# -*- coding: utf-8 -*-

# IS 6733
# Big Data Technology
# Fall 2018

# Project: Neural Networks to Predict Likelihood of Illness, Given Other Conditions

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# Importing the Keras libraries and packages
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers



# C:\Users\m730085\Google Drive\UTSA\Course 7 - IS 6733 Big Data Technology\BDT_Project

# Data Import and Typecasting

hccData = pd.read_csv('C:/Users/m730085/Google Drive/UTSA/Course 7 - IS 6733 Big Data Technology/BDT_Project/UPDATED DATA/ProjectData.csv', sep="|")
hcc2017 = hccData.loc[hccData['MMESource'] == 2017.0]
hcc2016 = hccData.loc[hccData['MMESource'] == 2016.0]
hcc2015 = hccData.loc[hccData['MMESource'] == 2015.0]
hcc2014 = hccData.loc[hccData['MMESource'] == 2014.0]

((hcc2017.loc[hcc2017['HCC22']==1]).shape[0])/hcc2017.shape[0]
((hcc2016.loc[hcc2016['HCC22']==1]).shape[0])/hcc2016.shape[0]
((hcc2015.loc[hcc2015['HCC22']==1]).shape[0])/hcc2015.shape[0]
((hcc2014.loc[hcc2014['HCC22']==1]).shape[0])/hcc2014.shape[0]

#Converting all data to float ----------------------------------------------------------------
# Creating matrix of features (X) 
# Creating matrix of target variable (Y)


#Converting all data to float
hcc2016[['Age',	'Gender',	'IS_HOSPICE_FLAG',	'IS_SNP_FLAG',	'HCC1',	'HCC2',	'HCC6',	'HCC8',	'HCC9',	'HCC10',	'HCC11',	'HCC12',	'HCC17',	'HCC18',	
         'HCC19',	'HCC21',	'HCC22',	'HCC23',	'HCC27',	'HCC28',	'HCC29',	'HCC33',	'HCC34',	'HCC35',	'HCC39',	'HCC40',	'HCC46',	
         'HCC47',	'HCC48',	'HCC54',	'HCC55',	'HCC57',	'HCC58',	'HCC70',	'HCC71',	'HCC72',	'HCC73',	'HCC74',	'HCC75',	'HCC76',	
         'HCC77',	'HCC78',	'HCC79',	'HCC80',	'HCC82',	'HCC83',	'HCC84',	'HCC85',	'HCC86',	'HCC87',	'HCC88',	'HCC96',	'HCC99',	
         'HCC100',	'HCC103',	'HCC104',	'HCC106',	'HCC107',	'HCC108',	'HCC110',	'HCC111',	'HCC112',	'HCC114',	'HCC115',	'HCC122',	'HCC124',	
         'HCC134',	'HCC135',	'HCC136',	'HCC137',	'HCC157',	'HCC158',	'HCC161',	'HCC162',	'HCC166',	'HCC167',	'HCC169',	'HCC170',	'HCC173',	
         'HCC176',	'HCC186',	'HCC188',	'HCC189',	'HCC138'
]] = hcc2016[['Age',	'Gender',	'IS_HOSPICE_FLAG',	'IS_SNP_FLAG',	'HCC1',	'HCC2',	'HCC6',	'HCC8',	'HCC9',	'HCC10',	'HCC11',	'HCC12',	'HCC17',	'HCC18',	
         'HCC19',	'HCC21',	'HCC22',	'HCC23',	'HCC27',	'HCC28',	'HCC29',	'HCC33',	'HCC34',	'HCC35',	'HCC39',	'HCC40',	'HCC46',	
         'HCC47',	'HCC48',	'HCC54',	'HCC55',	'HCC57',	'HCC58',	'HCC70',	'HCC71',	'HCC72',	'HCC73',	'HCC74',	'HCC75',	'HCC76',	
         'HCC77',	'HCC78',	'HCC79',	'HCC80',	'HCC82',	'HCC83',	'HCC84',	'HCC85',	'HCC86',	'HCC87',	'HCC88',	'HCC96',	'HCC99',	
         'HCC100',	'HCC103',	'HCC104',	'HCC106',	'HCC107',	'HCC108',	'HCC110',	'HCC111',	'HCC112',	'HCC114',	'HCC115',	'HCC122',	'HCC124',	
         'HCC134',	'HCC135',	'HCC136',	'HCC137',	'HCC157',	'HCC158',	'HCC161',	'HCC162',	'HCC166',	'HCC167',	'HCC169',	'HCC170',	'HCC173',	
         'HCC176',	'HCC186',	'HCC188',	'HCC189',	'HCC138']].astype("float")
hcc2016.dtypes
#hcc2016['HCC22'] = hcc2016['HCC22'].astype("category")
hcc2016['HCC22'].describe()

X = hcc2016.drop(['MMESource','BISource','ID','HCC22'], 1)
y = hcc2016.loc[:, ['HCC22']]
y.to_csv('y.csv')




# Creating Validation matrix of features (X) 
# Creating Validation matrix of target variable (Y)

#Converting all data to float
hcc2017[['Age',	'Gender',	'IS_HOSPICE_FLAG',	'IS_SNP_FLAG',	'HCC1',	'HCC2',	'HCC6',	'HCC8',	'HCC9',	'HCC10',	'HCC11',	'HCC12',	'HCC17',	'HCC18',	
         'HCC19',	'HCC21',	'HCC22',	'HCC23',	'HCC27',	'HCC28',	'HCC29',	'HCC33',	'HCC34',	'HCC35',	'HCC39',	'HCC40',	'HCC46',	
         'HCC47',	'HCC48',	'HCC54',	'HCC55',	'HCC57',	'HCC58',	'HCC70',	'HCC71',	'HCC72',	'HCC73',	'HCC74',	'HCC75',	'HCC76',	
         'HCC77',	'HCC78',	'HCC79',	'HCC80',	'HCC82',	'HCC83',	'HCC84',	'HCC85',	'HCC86',	'HCC87',	'HCC88',	'HCC96',	'HCC99',	
         'HCC100',	'HCC103',	'HCC104',	'HCC106',	'HCC107',	'HCC108',	'HCC110',	'HCC111',	'HCC112',	'HCC114',	'HCC115',	'HCC122',	'HCC124',	
         'HCC134',	'HCC135',	'HCC136',	'HCC137',	'HCC157',	'HCC158',	'HCC161',	'HCC162',	'HCC166',	'HCC167',	'HCC169',	'HCC170',	'HCC173',	
         'HCC176',	'HCC186',	'HCC188',	'HCC189',	'HCC138'
]] = hcc2017[['Age',	'Gender',	'IS_HOSPICE_FLAG',	'IS_SNP_FLAG',	'HCC1',	'HCC2',	'HCC6',	'HCC8',	'HCC9',	'HCC10',	'HCC11',	'HCC12',	'HCC17',	'HCC18',	
         'HCC19',	'HCC21',	'HCC22',	'HCC23',	'HCC27',	'HCC28',	'HCC29',	'HCC33',	'HCC34',	'HCC35',	'HCC39',	'HCC40',	'HCC46',	
         'HCC47',	'HCC48',	'HCC54',	'HCC55',	'HCC57',	'HCC58',	'HCC70',	'HCC71',	'HCC72',	'HCC73',	'HCC74',	'HCC75',	'HCC76',	
         'HCC77',	'HCC78',	'HCC79',	'HCC80',	'HCC82',	'HCC83',	'HCC84',	'HCC85',	'HCC86',	'HCC87',	'HCC88',	'HCC96',	'HCC99',	
         'HCC100',	'HCC103',	'HCC104',	'HCC106',	'HCC107',	'HCC108',	'HCC110',	'HCC111',	'HCC112',	'HCC114',	'HCC115',	'HCC122',	'HCC124',	
         'HCC134',	'HCC135',	'HCC136',	'HCC137',	'HCC157',	'HCC158',	'HCC161',	'HCC162',	'HCC166',	'HCC167',	'HCC169',	'HCC170',	'HCC173',	
         'HCC176',	'HCC186',	'HCC188',	'HCC189',	'HCC138']].astype("float")
hcc2017.dtypes
#hcc2015['HCC22'] = hcc2015['HCC22'].astype("category")

X_test = hcc2017.drop(['MMESource','BISource','ID','HCC22'], 1)
y_test = hcc2017.loc[:, ['HCC22']]
y_test.describe()
y_test.to_csv('ytest.csv')




# Train/Test Split ---------------------------------------------------------------

X_train, X_Val, y_train, y_Val = train_test_split(X, y, test_size = 0.3)


# Build Initial model ----------------------------------------------------------------------

#Initializing Neural Network
classifier = Sequential()
# Adding layers
classifier.add(Dense(units = 100, activation = 'relu', input_dim = X.shape[1]))
classifier.add(Dense(units = 50, activation = 'relu'))
classifier.add(Dense(units = 25, activation = 'relu'))
classifier.add(Dense(units = 15, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.summary()


classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

learning_rate = 0.01
epochs_value = 500
batch_size_value = 1000


# Train and Eval the Model -------------------------------------------------------------------------

HCCModel = classifier.fit(X_train, 
                          y_train, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          validation_data=(X_Val, y_Val))
                          #validation_data=(X_test, y_test))



classifier.save('HCCInitial.keras', overwrite=True)  

score = classifier.evaluate(X_Val, y_Val)
#classifier.evaluate(X_test, y_test)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])



# Plot the Loss curves ------------------------------------------------------------------------------

plt.plot(range(1,epochs_value+1), HCCModel.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()


# Prediction on Test set --------------------------------------------------------------

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.9)


#  Confusion Matrix ----------------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)
print(cm)

#Set up for PPV, not sensitivity
cm[1,1]/(cm[0,1]+cm[1,1])










# Model 2 - fewer epochs ---------------------------------------------------------

learning_rate = 0.01
epochs_value = 100
batch_size_value = 1000

HCCModel2 = classifier.fit(X_train, 
                          y_train, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          validation_data=(X_Val, y_Val))
                          #validation_data=(X_test, y_test))

classifier.save('HCCShort.keras', overwrite=True)

score2 = classifier.evaluate(X_Val, y_Val)
#classifier.evaluate(X_test, y_test)
print('Validation loss:', score2[0])
print('Validation accuracy:', score2[1])

plt.plot(range(1,epochs_value+1), HCCModel2.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel2.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score2[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred2 = classifier.predict(X_test)
y_pred2 = (y_pred2 > 0.9)

cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

#Set up for PPV, not sensitivity
cm2[1,1]/(cm2[0,1]+cm2[1,1])

# -----------------------------------------------------------------------------------------










# Model 3: Change Learning Rate (.1) ------------------------------------------------------

learning_rate = 0.1
epochs_value = 100
batch_size_value = 1000

HCCModel3 = classifier.fit(X_train, 
                          y_train, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          validation_data=(X_Val, y_Val))
                          #validation_data=(X_test, y_test))

classifier.save('HCCShortLearn1.keras', overwrite=True)

score3 = classifier.evaluate(X_Val, y_Val)
#classifier.evaluate(X_test, y_test)
print('Validation loss:', score3[0])
print('Validation accuracy:', score3[1])

plt.plot(range(1,epochs_value+1), HCCModel3.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel3.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score3[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred3 = classifier.predict(X_test)
y_pred3 = (y_pred3 > 0.9)

cm3 = confusion_matrix(y_test, y_pred3)
print(cm3)

#Set up for PPV, not sensitivity
print(cm3[1,1]/(cm3[0,1]+cm3[1,1]))





# Model 4: Reduce # Learned Parameters ---------------------------------------------------------------

#Initializing Neural Network
classifier4 = Sequential()
# Adding the layers and the first hidden layer
classifier4.add(Dense(units = 50, activation = 'relu', input_dim = X.shape[1]))
classifier4.add(Dense(units = 25, activation = 'relu'))
classifier4.add(Dense(units = 15, activation = 'relu'))
classifier4.add(Dense(units = 1, activation = 'sigmoid'))
classifier4.summary()


classifier4.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

learning_rate = 0.01
epochs_value = 500
batch_size_value = 1000

HCCModel4 = classifier4.fit(X_train, 
                          y_train, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          validation_data=(X_Val, y_Val))
                          #validation_data=(X_test, y_test))

classifier4.save('HCCSmall.keras', overwrite=True)

score4 = classifier4.evaluate(X_Val, y_Val)
#classifier.evaluate(X_test, y_test)
print('Validation loss:', score4[0])
print('Validation accuracy:', score4[1])

plt.plot(range(1,epochs_value+1), HCCModel4.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel4.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score4[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred4 = classifier4.predict(X_test)
y_pred4 = (y_pred4 > 0.9)

cm4 = confusion_matrix(y_test, y_pred4)
print(cm4)

#Set up for PPV, not sensitivity
print(cm4[1,1]/(cm4[0,1]+cm4[1,1]))





# Model 5: Reduce Epochs on Smaller Model -----------------------------------------------------

learning_rate = 0.01
epochs_value = 150
batch_size_value = 1000

HCCModel5 = classifier4.fit(X_train, 
                          y_train, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          validation_data=(X_Val, y_Val))
                          #validation_data=(X_test, y_test))

classifier4.save('HCCSmallShort.keras', overwrite=True)

score5 = classifier4.evaluate(X_Val, y_Val)
#classifier.evaluate(X_test, y_test)
print('Validation loss:', score5[0])
print('Validation accuracy:', score5[1])

plt.plot(range(1,epochs_value+1), HCCModel5.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel5.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score5[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred5 = classifier4.predict(X_test)
y_pred5 = (y_pred5 > 0.9)

cm5 = confusion_matrix(y_test, y_pred5)
print(cm5)

#Set up for PPV, not sensitivity
print(cm5[1,1]/(cm5[0,1]+cm5[1,1]))



# Model 6: Increase # Learned Parameters ----------------------------------------------------------

#Initializing Neural Network
classifier6 = Sequential()
# Adding the layers and the first hidden layer
classifier6.add(Dense(units = 500, activation = 'relu', input_dim = X.shape[1]))
classifier6.add(Dense(units = 250, activation = 'relu'))
classifier6.add(Dense(units = 125, activation = 'relu'))
classifier6.add(Dense(units = 60, activation = 'relu'))
classifier6.add(Dense(units = 30, activation = 'relu'))
classifier6.add(Dense(units = 15, activation = 'relu'))
classifier6.add(Dense(units = 1, activation = 'sigmoid'))
classifier6.summary()


classifier6.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

learning_rate = 0.01
epochs_value = 500
batch_size_value = 1000

HCCModel6 = classifier6.fit(X_train, 
                          y_train, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          validation_data=(X_Val, y_Val))
                          #validation_data=(X_test, y_test))

classifier6.save('HCCLarge.keras', overwrite=True)

score6 = classifier6.evaluate(X_Val, y_Val)
#classifier.evaluate(X_test, y_test)
print('Validation loss:', score6[0])
print('Validation accuracy:', score6[1])

plt.plot(range(1,epochs_value+1), HCCModel6.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel6.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score6[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred6 = classifier6.predict(X_test)
y_pred6 = (y_pred4 > 0.9)

cm6 = confusion_matrix(y_test, y_pred6)
print(cm4)

#Set up for PPV, not sensitivity
print(cm6[1,1]/(cm6[0,1]+cm6[1,1]))








# Model 7: Regularize L1 ------------------------------------------------------------------------

learning_rate = 0.01
epochs_value = 500
batch_size_value = 1000

#Initializing Neural Network
classifier7 = Sequential()
# Adding the layers and the first hidden layer
classifier7.add(Dense(units = 100, kernel_regularizer = regularizers.l1(learning_rate), activation = 'relu', input_dim = X.shape[1]))
classifier7.add(Dense(units = 50, kernel_regularizer = regularizers.l1(learning_rate), activation = 'relu', input_dim = X.shape[1]))
classifier7.add(Dense(units = 25, kernel_regularizer = regularizers.l1(learning_rate),  activation = 'relu'))
classifier7.add(Dense(units = 15, kernel_regularizer = regularizers.l1(learning_rate),  activation = 'relu'))
classifier7.add(Dense(units = 1, kernel_regularizer = regularizers.l1(learning_rate), activation = 'sigmoid'))
classifier7.summary()


classifier7.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

HCCModel7 = classifier7.fit(X_train, 
                          y_train, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          #validation_data=(X_Val, y_Val))
                          validation_data=(X_test, y_test))

classifier7.save('HCCRegL1.keras', overwrite=True)

score7 = classifier7.evaluate(X_Val, y_Val)
#classifier.evaluate(X_test, y_test)
print('Validation loss:', score7[0])
print('Validation accuracy:', score7[1])

plt.plot(range(1,epochs_value+1), HCCModel7.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel7.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score7[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred7 = classifier7.predict(X_test)
y_pred7 = (y_pred7 > 0.9)

cm7 = confusion_matrix(y_test, y_pred7)
print(cm7)

#Set up for PPV, not sensitivity
print(cm7[1,1]/(cm7[0,1]+cm7[1,1]))









# Model 8: Regularize L2 --------------------------------------------------------------

learning_rate = 0.01
epochs_value = 500
batch_size_value = 1000

#Initializing Neural Network
classifier8 = Sequential()
# Adding the layers and the first hidden layer
classifier8.add(Dense(units = 100, kernel_regularizer = regularizers.l2(learning_rate), activation = 'relu', input_dim = X.shape[1]))
classifier8.add(Dense(units = 50, kernel_regularizer = regularizers.l2(learning_rate), activation = 'relu', input_dim = X.shape[1]))
classifier8.add(Dense(units = 25, kernel_regularizer = regularizers.l2(learning_rate),  activation = 'relu'))
classifier8.add(Dense(units = 15, kernel_regularizer = regularizers.l2(learning_rate),  activation = 'relu'))
classifier8.add(Dense(units = 1, kernel_regularizer = regularizers.l2(learning_rate), activation = 'sigmoid'))
classifier8.summary()


classifier8.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

HCCModel8 = classifier8.fit(X_train, 
                          y_train, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          #validation_data=(X_Val, y_Val))
                          validation_data=(X_test, y_test))

classifier8.save('HCCRegL2.keras', overwrite=True)

score8 = classifier8.evaluate(X_Val, y_Val)
#classifier.evaluate(X_test, y_test)
print('Validation loss:', score8[0])
print('Validation accuracy:', score8[1])

plt.plot(range(1,epochs_value+1), HCCModel8.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel8.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score8[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred8 = classifier8.predict(X_test)
y_pred8 = (y_pred8 > 0.9)

cm8 = confusion_matrix(y_test, y_pred7)
print(cm7)

#Set up for PPV, not sensitivity
print(cm8[1,1]/(cm8[0,1]+cm8[1,1]))



# Model 9: Remove All-Zero rows from Dataset, run Model 1 on edited set -------------------------

hcc2017adj = hcc2017[hcc2017.iloc[:,6:].sum(axis=1) > 0]
hcc2016adj = hcc2016[hcc2016.iloc[:,6:].sum(axis=1) > 0]

X_adj = hcc2016adj.drop(['MMESource','BISource','ID','HCC22'], 1)
y_adj = hcc2016adj.loc[:, ['HCC22']]
X_test_adj = hcc2017adj.drop(['MMESource','BISource','ID','HCC22'], 1)
y_test_adj = hcc2017adj.loc[:, ['HCC22']]

X_train_adj, X_Val_adj, y_train_adj, y_Val_adj = train_test_split(X_adj, y_adj, test_size = 0.3)


HCCModel9 = classifier.fit(X_train_adj, 
                          y_train_adj, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          validation_data=(X_Val_adj, y_Val_adj))
                          #validation_data=(X_test, y_test))



classifier.save('HCCInitialAdj.keras', overwrite=True)  

score9 = classifier.evaluate(X_Val_adj, y_Val_adj)
print('Validation loss:', score9[0])
print('Validation accuracy:', score9[1])

plt.plot(range(1,epochs_value+1), HCCModel9.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel9.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score9[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred9 = classifier.predict(X_test_adj)
y_pred9 = (y_pred9 > 0.9)


cm9 = confusion_matrix(y_test_adj, y_pred9)
print(cm9)

print(cm9[1,1]/(cm9[0,1]+cm9[1,1]))






# Model 10: Small Epoch count on adjusted set --------------------------------------------------


learning_rate = 0.01
epochs_value = 100
batch_size_value = 1000

HCCModel10 = classifier.fit(X_train_adj, 
                          y_train_adj, 
                          batch_size=batch_size_value, 
                          epochs=epochs_value,
                          validation_data=(X_Val_adj, y_Val_adj))
                          #validation_data=(X_test, y_test))



classifier.save('HCCSmallAdj.keras', overwrite=True)  

score10 = classifier.evaluate(X_Val_adj, y_Val_adj)
print('Validation loss:', score10[0])
print('Validation accuracy:', score10[1])

plt.plot(range(1,epochs_value+1), HCCModel10.history['loss'], 'r+', label='training loss')
plt.plot(range(1,epochs_value+1), HCCModel10.history['val_loss'], 'bo', label='evaluation loss')
plt.xlabel('Epochs\n\n Accuracy: '+str(score10[1]))
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred10 = classifier.predict(X_test_adj)
y_pred10 = (y_pred10 > 0.9)


cm10 = confusion_matrix(y_test_adj, y_pred10)
print(cm10)

print(cm10[1,1]/(cm10[0,1]+cm10[1,1]))








