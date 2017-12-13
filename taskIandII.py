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

#print x.describe(percentiles=[0.25, 0.9])
#print y.describe(percentiles=[0.25, 0.9])

#print x[x['X..memused'] > 80].count()['X..memused']
#print x[x['sum_intr.s'] > 18000].sum()['tcpsck'] / x[x['sum_intr.s'] > 18000].count()['tcpsck']
#print x[x['all_..idle'] < 20].min()['X..memused']

#plt.plot(x['TimeStamp'], x['all_..idle'])
#plt.plot(x['TimeStamp'], x['X..memused'])
#plt.show()

#plt.hist(x['all_..idle'])
#plt.show()

#plt.hist(x['X..memused'])
#plt.show()

#plt.boxplot(x['all_..idle'])
#plt.show()

#plt.boxplot(x['X..memused'])
#plt.show()

#kde = gaussian_kde(x['all_..idle'])
#dist_space = linspace( min(x['all_..idle']), max(x['all_..idle']), 100 )
#plt.plot( dist_space, kde(dist_space) )
#plt.show()

#kde = gaussian_kde(x['X..memused'])
#dist_space = linspace( min(x['X..memused']), max(x['X..memused']), 100 )
#plt.plot( dist_space, kde(dist_space) )
#plt.show()
