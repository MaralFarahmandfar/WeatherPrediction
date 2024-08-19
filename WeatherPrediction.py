import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM , Dense,Dropout
from keras.src.models import Sequential
import math

#بارگذاری اطلاعات
data = pd.read_csv("E:\.vscode\weather prediction\datas.csv")

#تبدیل به دیتاتایپ مناسب 
data['time']=pd.to_datetime(data['time'])
data['temperature_2m_mean']=pd.to_numeric(data['temperature_2m_mean'])

#تاریخ شروع و پایان مجموعه آموزش و آزمون
train_start=pd.Timestamp('2014-01-01')
train_end=pd.Timestamp('2022-12-31')
test_start=pd.Timestamp('2023-01-01')
test_end=pd.Timestamp('2024-01-01')

#جداسازی مجموعه آموزش
condition1=(data['time']>=train_start)&(data['time']<=train_end)
if condition1.any():
    Train=data[condition1]

#جداسازی مجموعه آزمون
condition2=(data['time']>=test_start)&(data['time']<=test_end)
if condition2.any():
    Test=data[condition2]

temp_train=Train[['temperature_2m_mean']]
temp_test=Test[['temperature_2m_mean']]

#نرمال سازی داده ها
scaler=MinMaxScaler(feature_range=(0,1))
train_scaled=scaler.fit_transform(temp_train)
test_scaled=scaler.transform(temp_test)

#ساخت داده ها با گام زمانی
def create_dataset(dataset,look_back=1):
    X_data,y_data=[],[]
    for i in range(len(dataset)-look_back-1):
        X_data.append(dataset[i:(i+look_back),0])
        y_data.append(dataset[i+look_back,0])
    return np.array(X_data), np.array(y_data)

look_back=100
trainX,trainY=create_dataset(train_scaled,look_back)
testX,testY=create_dataset(test_scaled,look_back)

#reshape input to be [samples,time steps,features]
trainX=np.reshape(trainX,(trainX.shape[0],trainX.shape[1],1))
testX=np.reshape(testX,(testX.shape[0],testX.shape[1],1))

#LSTMساخت شبکه عصبی 
model=Sequential()
model.add(LSTM(256,return_sequences=True,input_shape=(look_back,1)))
model.add(Dropout(0.3))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dense(32))
model.add(Dense(1))

#کامپایل مدل
model.compile(loss='mean_squared_error',optimizer='adam')

# تنظیم Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# آموزش مدل با Early Stopping
model.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY), callbacks=[early_stopping])

#پیش بینی
trainpredict=model.predict(trainX)
testpredict=model.predict(testX)

#ارزیابی مدل و محاسبه خطا
trainpredict=scaler.inverse_transform(trainpredict)
trainY=scaler.inverse_transform(trainY.reshape(-1, 1))
testpredict=scaler.inverse_transform(testpredict)
testY=scaler.inverse_transform(testY.reshape(-1, 1))

trainScore = math.sqrt(mean_squared_error(trainY, trainpredict))
print("train score: %.2f RMSE"% (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testpredict))
print("test score: %.2f RMSE"% (testScore))

#نمایش روی نمودار
trainPredictPlot = np.empty((len(temp_train), 1))
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(trainpredict)+look_back,:]=trainpredict

testPredictPlot = np.empty((len(data['temperature_2m_mean']), 1))
testPredictPlot[:,:]=np.nan
testPredictPlot[len(trainpredict)+(look_back*2)+1:3652,:]=testpredict

plt.plot(data['temperature_2m_mean'],label='Data',linestyle='--')
plt.plot(trainPredictPlot,label='Train Predictions')
plt.plot(testPredictPlot,label='Test Predictions')
plt.title('Weather Prediction')
plt.xlabel('Day')
plt.ylabel('Temperature')

#پیش بینی روز های آینده
lastdays=trainX[-6].reshape(1,look_back,1) 
futureDays=365 
prediction=[]
noise_factor=0.01

for _ in range(futureDays): 
    predictFuture=model.predict(lastdays) 
    predictFuture += np.random.normal(scale=noise_factor) 
    prediction.append(predictFuture[0,0]) 
    lastdays = np.append(lastdays[:,1:,:], predictFuture.reshape(1,1,1),axis=1) 
    
prediction=np.array(prediction) 
prediction=scaler.inverse_transform(prediction.reshape(-1,1)) 
print(prediction)
futurePredictPlot = np.empty((4018, 1)) 
futurePredictPlot[:,:]=np.nan
futurePredictPlot[3652:4017,:]=prediction

plt.plot(futurePredictPlot,label='Future Predictions')
plt.legend() 
plt.show() 