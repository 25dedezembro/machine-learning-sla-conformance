# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace

# importando dados e gerando duas matrizes
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

# fim da importação de X e Y

# Task 1 exercício 1 
# 1. Compute the following statistics for each component of X and Y: 
# mean, maximum, minimum, 25th
# percentile, 90th percentile, and standard deviation.

print "Task I 1: \n"

print "Describing X: \n"
print x.describe(percentiles=[0.25, 0.9])

print "\nDescribing Y: \n"
print y.describe(percentiles=[0.25, 0.9])

# Task 1 exercício 2
# 2. Compute the following quantities:

# (a) the number of observations with memory usage larger than 80%;
print "\nTask I 2. (a): \n"
print x[x['X..memused'] > 80].count()['X..memused']

# (b) the average number of used TCP sockets for observations with more than 18000 interrupts/sec;
print "\nTask I 2. (b): \n"
print x[x['sum_intr.s'] > 18000].sum()['tcpsck'] / x[x['sum_intr.s'] > 18000].count()['tcpsck']

# (c) the minimum memory utilization for observations with CPU idle time lower than 20%.
print "\nTask I 2. (c): \n"
print x[x['all_..idle'] < 20].min()['X..memused']

# Task 1 exercício 3 
# 3. Produce the following plots:

# (a) Time series of percentage of idle CPU and of used memory (both in a single plot);

print "\nTask I 3. (a): \n"

plt.xlabel('Tempo')
plt.ylabel('Porcentagem')
plt.plot(x['TimeStamp'], x['all_..idle'],label='idle CPU')
plt.plot(x['TimeStamp'], x['X..memused'],label='Used memory')
legend = plt.legend(loc='best', shadow=True)
plt.show()

# (b) Density plots, histograms, and box plots of idle CPU and of used memory.


print "\nTask I 3. (b): \n"

# histogramas
plt.hist(x['all_..idle'])
plt.xlabel('idle CPU (%)')
plt.ylabel('Frequencia')
plt.show()

plt.hist(x['X..memused'])
plt.xlabel('used memory (%)')
plt.ylabel('Frequencia')
plt.show()

# boxplot 
boxplot_task3 = [ x['all_..idle'], x['X..memused']]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(boxplot_task3)
plt.xlabel('idle CPU (1) e used memory (2)')
plt.ylabel('Porcentagem')
plt.show()

# densidade
kde = gaussian_kde(x['all_..idle'])
dist_space = linspace( min(x['all_..idle']), max(x['all_..idle']), 100 )
plt.plot( dist_space, kde(dist_space) )
plt.xlabel('idle CPU (%)')
plt.ylabel('Densidade')
plt.show()
kde = gaussian_kde(x['X..memused'])
dist_space = linspace( min(x['X..memused']), max(x['X..memused']), 100 )
plt.plot( dist_space, kde(dist_space) )
plt.xlabel('used memory (%)')
plt.ylabel('Densidade')
plt.show()
