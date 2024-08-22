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
path=input("Enter path of the datas: ")
path2=input("Enter data file name with .csv: ")
path=path+"/"+path2
data = pd.read_csv(path)
print(data.columns)

date=input("Enter the date column name: ")
temp=input("Enter the temperature column name: ")

#تبدیل به دیتاتایپ مناسب 
data[date]=pd.to_datetime(data[date])
data[temp]=pd.to_numeric(data[temp])

#تاریخ شروع و پایان مجموعه آموزش و آزمون
train_start=pd.Timestamp(input("Enter date of the train_start: "))
train_end=pd.Timestamp(input("Enter date of the train_end: "))
test_start=pd.Timestamp(input("Enter date of the test_start: "))
test_end=pd.Timestamp(input("Enter date of the test_end: "))

#جداسازی مجموعه آموزش
condition1=(data[date]>=train_start)&(data[date]<=train_end)
if condition1.any():
    Train=data[condition1]

#جداسازی مجموعه آزمون
condition2=(data[date]>=test_start)&(data[date]<=test_end)
if condition2.any():
    Test=data[condition2]

temp_train=Train[[temp]]
temp_test=Test[[temp]]

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

if len(data)>=1000 and len(data)<2000:
    look_back=60
elif len(data)>=2000 and len(data)<3000:
    look_back=100
elif len(data)>=3000 and len(data)<=4000:
    look_back=100

trainX,trainY=create_dataset(train_scaled,look_back)
testX,testY=create_dataset(test_scaled,look_back)

#reshape input to be [samples,time steps,features]
trainX=np.reshape(trainX,(trainX.shape[0],trainX.shape[1],1))
testX=np.reshape(testX,(testX.shape[0],testX.shape[1],1))

#LSTMساخت شبکه عصبی 
if len(data)>=1000 and len(data)<2000:

    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(look_back,1)))
    model.add(LSTM(50))

elif len(data)>=2000 and len(data)<3000:
    model=Sequential()
    model.add(LSTM(128,return_sequences=True,input_shape=(look_back,1)))
    model.add(LSTM(64))
    model.add(Dense(25))

elif len(data)>=3000 and len(data)<=4000:
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
if len(data)>=1000 and len(data)<2000:
    model.fit(trainX, trainY, epochs=30, batch_size=15, validation_data=(testX, testY), callbacks=[early_stopping])
elif len(data)>=2000 and len(data)<3000:
    model.fit(trainX, trainY, epochs=30, batch_size=15, validation_data=(testX, testY), callbacks=[early_stopping])
elif len(data)>=3000 and len(data)<=4000:
    model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[early_stopping])

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
trainPredictPlot = np.empty((len(data[temp]), 1))
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(trainpredict)+look_back,:]=trainpredict

testPredictPlot = np.empty((len(data[temp]), 1))
testPredictPlot[:,:]=np.nan
testPredictPlot[len(trainpredict)+(look_back*2)+1:len(data[temp])-1,:]=testpredict

plt.plot(data[date],data[temp],label='Data',linestyle='--')
plt.plot(data[date],trainPredictPlot,label='Train Predictions')
plt.plot(data[date],testPredictPlot,label='Test Predictions')
plt.title('Weather Prediction')
plt.xlabel('Date')
plt.ylabel('Temperature')

#پیش بینی روز های آینده

futureDays=int(input("Enter the number of days ahead for the forcast: ")) 
lastdays=train_scaled[-look_back:].reshape(1,look_back,1) 
prediction=[]
noise_factor=0.005

for _ in range(futureDays): 
    predictFuture=model.predict(lastdays) 
    predictFuture += np.random.normal(scale=noise_factor) 
    prediction.append(predictFuture[0,0]) 
    lastdays = np.append(lastdays[:,1:,:], predictFuture.reshape(1,1,1),axis=1) 
    
prediction=np.array(prediction) 
prediction=scaler.inverse_transform(prediction.reshape(-1,1)) 

last_date = data[date].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=futureDays+1, freq='D')[1:]

plt.plot(future_dates, prediction, label='Future Predictions')
plt.gcf().autofmt_xdate()
plt.legend() 
plt.show() 
