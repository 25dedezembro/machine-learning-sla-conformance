import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace

input_file_x = "X.csv"
input_file_y = "Y.csv"

x = pd.read_csv(input_file_x, header = 0)
y = pd.read_csv(input_file_y, header = 0)

original_headers_x = list(x.columns.values)
x = x._get_numeric_data()
numeric_headers_x = list(x.columns.values)

original_headers_y = list(y.columns.values)
y = y._get_numeric_data()
numeric_headers_y = list(y.columns.values)

numpy_array_x = x.as_matrix()
numpy_array_y = y.as_matrix()

print x.describe(percentiles=[0.25, 0.9])
print y.describe(percentiles=[0.25, 0.9])

print x[x['X..memused'] > 80].count()['X..memused']

print x[x['sum_intr.s'] > 18000].sum()['tcpsck'] / x[x['sum_intr.s'] > 18000].count()['tcpsck']
print x[x['all_..idle'] < 20].min()['X..memused']

plt.plot(x['TimeStamp'], x['all_..idle'])
plt.plot(x['TimeStamp'], x['X..memused'])
plt.show()

plt.hist(x['all_..idle'])
plt.show()

plt.hist(x['X..memused'])
plt.show()

plt.boxplot(x['all_..idle'])
plt.show()

plt.boxplot(x['X..memused'])
plt.show()

kde = gaussian_kde(x['all_..idle'])
dist_space = linspace( min(x['all_..idle']), max(x['all_..idle']), 100 )
plt.plot( dist_space, kde(dist_space) )
plt.show()

kde = gaussian_kde(x['X..memused'])
dist_space = linspace( min(x['X..memused']), max(x['X..memused']), 100 )
plt.plot( dist_space, kde(dist_space) )
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

input_file_x = "X.csv"
input_file_y = "Y.csv"

x = pd.read_csv(input_file_x, header = 0)
y = pd.read_csv(input_file_y, header = 0)

original_headers_x = list(x.columns.values)
x = x._get_numeric_data()
numeric_headers_x = list(x.columns.values)

original_headers_y = list(y.columns.values)
y = y._get_numeric_data()
numeric_headers_y = list(y.columns.values)

numpy_array_x = x.as_matrix()
numpy_array_y = y.as_matrix()

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(x, y, test_size=0.30)

lm = LinearRegression()

X_train = X_train_full.drop('TimeStamp', axis = 1)
y_train = y_train_full.drop('TimeStamp', axis = 1)
X_test = X_test_full.drop('TimeStamp', axis = 1)
y_test = y_test_full.drop('TimeStamp', axis = 1)

lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

print ('Coefficients: \n', lm.coef_)

print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, y_pred))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.plot(y_test)
plt.plot(y_pred)
plt.show()

plt.hist(y_test_full['DispFrames'])
plt.show()

kde = gaussian_kde(y_test_full['DispFrames'])
dist_space = linspace( min(y_test_full['DispFrames']), max(y_test_full['DispFrames']), 100 )
plt.plot( dist_space, kde(dist_space) )
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

e50 = []
for i in range(1, 50):
  X_train_full50, X_test_full50, y_train_full50, y_test_full50 = train_test_split(X_train_full, y_train_full, train_size=50)
  lm = LinearRegression()
  X_train50 = X_train_full50.drop('TimeStamp', axis = 1)
  y_train50 = y_train_full50.drop('TimeStamp', axis = 1)
  X_test50 = X_test_full50.drop('TimeStamp', axis = 1)
  y_test50 = y_test_full50.drop('TimeStamp', axis = 1)
  lm.fit(X_train50, y_train50)
  y_pred = lm.predict(X_test)
  e50 = e50 + [mean_absolute_error(y_test, y_pred)]


e100 = []
for i in range(1, 50):
  X_train_full100, X_test_full100, y_train_full100, y_test_full100 = train_test_split(X_train_full, y_train_full, train_size=100)
  lm = LinearRegression()
  X_train100 = X_train_full100.drop('TimeStamp', axis = 1)
  y_train100 = y_train_full100.drop('TimeStamp', axis = 1)
  X_test100 = X_test_full100.drop('TimeStamp', axis = 1)
  y_test100 = y_test_full100.drop('TimeStamp', axis = 1)
  lm.fit(X_train100, y_train100)
  y_pred = lm.predict(X_test)
  e100 = e100 + [mean_absolute_error(y_test, y_pred)]

print e100

e200 = []
for i in range(1, 50):
  X_train_full200, X_test_full200, y_train_full200, y_test_full200 = train_test_split(X_train_full, y_train_full, train_size=200)
  lm = LinearRegression()
  X_train200 = X_train_full200.drop('TimeStamp', axis = 1)
  y_train200 = y_train_full200.drop('TimeStamp', axis = 1)
  X_test200 = X_test_full200.drop('TimeStamp', axis = 1)
  y_test200 = y_test_full200.drop('TimeStamp', axis = 1)
  lm.fit(X_train200, y_train200)
  y_pred = lm.predict(X_test)
  e200 = e200 + [mean_absolute_error(y_test, y_pred)]

print e200


e500 = []
for i in range(1, 50):
  X_train_full500, X_test_full500, y_train_full500, y_test_full500 = train_test_split(X_train_full, y_train_full, train_size=500)
  lm = LinearRegression()
  X_train500 = X_train_full500.drop('TimeStamp', axis = 1)
  y_train500 = y_train_full500.drop('TimeStamp', axis = 1)
  X_test500 = X_test_full500.drop('TimeStamp', axis = 1)
  y_test500 = y_test_full500.drop('TimeStamp', axis = 1)
  lm.fit(X_train500, y_train500)
  y_pred = lm.predict(X_test)
  e500 = e500 + [mean_absolute_error(y_test, y_pred)]

print e500


e1000 = []
for i in range(1, 50):
  X_train_full1000, X_test_full1000, y_train_full1000, y_test_full1000 = train_test_split(X_train_full, y_train_full, train_size=1000)
  lm = LinearRegression()
  X_train1000 = X_train_full1000.drop('TimeStamp', axis = 1)
  y_train1000 = y_train_full1000.drop('TimeStamp', axis = 1)
  X_test1000 = X_test_full1000.drop('TimeStamp', axis = 1)
  y_test1000 = y_test_full1000.drop('TimeStamp', axis = 1)
  lm.fit(X_train1000, y_train1000)
  y_pred = lm.predict(X_test)
  e1000 = e1000 + [mean_absolute_error(y_test, y_pred)]

print e1000


e2520 = []
for i in range(1, 50):
  lm = LinearRegression()
  X_train2520 = X_train_full.drop('TimeStamp', axis = 1)
  y_train2520 = y_train_full.drop('TimeStamp', axis = 1)
  X_test2520 = X_test_full.drop('TimeStamp', axis = 1)
  y_test2520 = y_test_full.drop('TimeStamp', axis = 1)
  lm.fit(X_train2520, y_train2520)
  y_pred = lm.predict(X_test)
  e2520 = e2520 + [mean_absolute_error(y_test, y_pred)]

print e2520


data_to_plot = [e50, e100, e200, e500, e1000, e2520]

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data_to_plot)

plt.show()



print sum(e50)/50

print sum(e100)/50

print sum(e200)/50

print sum(e500)/50

print sum(e1000)/50

print sum(e2520)/50
