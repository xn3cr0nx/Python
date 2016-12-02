import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


style.use('ggplot')

##############In this part are been made features
df = quandl.get("WIKI/GOOGL")	#it return a DataFrame

#we just care about these columns
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

#we define two new columns HL_PCT, PCT_change this way:
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


##############In this part are been made labels
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)	#fillna is used to fill the holes with the first value

#we try to predict data of 10% of the dataset
forecast_out = int(math.ceil(0.1*len(df)))	#math.ceil returns the ceiling of df as a float, the smallest integer value greater than or equal to df

df['label'] = df[forecast_col].shift(-forecast_out)	#Shift index by desired number of periods with an optional time freq, forecast_out is the period and is -31 in this case

#features = X, labels = y
X = np.array(df.drop(['label','Adj. Close'], 1))	#drop returns new object with labels in requested axis removed (1 is column)
X = preprocessing.scale(X)	#scales the data
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)	#returns object with labels on given axis omitted where alternately any or all of the data are missing
y = np.array(df['label'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)




# clf = LinearRegression(n_jobs = -1)
# #clf = svm.SVR(kernel = 'poly')	#embarassing
# clf.fit(X_train, y_train)
######Pickling
# with open('linearregression.pickle', 'wb') as f:	#we need to save the classifier to avoid the train every step
# 	pickle.dump(clf, f)
pickle_in = open('linearregression.pickle')	#once we saved, we don't need to delcare our clf
clf = pickle.load(pickle_in)
######
accuracy = clf.score(X_test, y_test)

#print accuracy




#X_lately getthe last 31 days
forecast_set = clf.predict(X_lately)	#this is the forecast of next 31 unknow values

#print forecast_set, accuracy, forecast_out
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.value / 1e9
one_day = 86400
next_unix = last_unix + one_day

print last_date, last_unix, next_unix

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#until now we say the stock's forecast on the plot using the regression algorithm 





#these are the first 6 videos
