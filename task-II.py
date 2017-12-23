# In[2]:


import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Função única para calcular o NMAE
def nmae(test, pred):
  Yi = pd.DataFrame(test)
  MYi = pd.DataFrame(pred)
   
  E = 0.0
  for m in range(0, len(Yi) - 1):
    E += abs((Yi.iloc[m]['DispFrames'] - MYi.iloc[m][0]))
    m += 1
  nmae_resultado = (E/m)/Yi.mean()
  return nmae_resultado

# Task II exercício 1
# 1. Evaluate the Accuracy of Service Metric Estimation

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(x, y, test_size=0.30)

lm = LinearRegression()

X_train = X_train_full.drop('TimeStamp', axis = 1)
y_train = y_train_full.drop('TimeStamp', axis = 1)
X_test = X_test_full.drop('TimeStamp', axis = 1)
y_test = y_test_full.drop('TimeStamp', axis = 1)

lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

# (a) Model Training - use linear regression to train a model M with the training set. Provide the
# coefficients (Θ 0 , ..., Θ 9 ) of your model M . (Θ 0 is the offset.)
print ('Coefficients: \n', lm.coef_)

cont = 0
coefficients = lm.coef_.tolist()
print coefficients

# (b) Accuracy of Model M - compute the estimation error of M over the test. 
# ...
# As a baseline for M , use a naive method which relies on Y values only.
# ...

print("NMAE: %.2f" % nmae(y_test, y_pred))

print y_pred.mean()

print('Variance score: %.2f' % r2_score(y_test, y_pred))

# (c) Produce a time series plot that shows both the measurements and the model estimations for M
# for the Video Frame Rate values in the test set (see example of such a plot in Figure 4(a) of [1]).
# Show also the prediction of the a naive method.

avg = y_test.mean()
mean = []
for i in range(1, len(y_test_full['TimeStamp']) +1 ):
  mean = mean + [avg]

plt.scatter(y_test_full['TimeStamp'], pd.DataFrame(y_test)['DispFrames'],25,'b','*', label='Test set')
plt.scatter(y_test_full['TimeStamp'], pd.DataFrame(y_pred)[0],25,'r','^', label='Measurements (Prediction set)')
plt.scatter(y_test_full['TimeStamp'], mean,25,'g','o', label='Naive')
plt.xlabel('Tempo')
plt.ylabel('Video Frame Rate')
legend = plt.legend(loc='best', shadow=True)
plt.show()

#print len(mean)
#print len(y_test_full['TimeStamp'])

# (d) Produce a density plot and a histogram for the Video Frame Rate values in the test set. Set the
# bin size of the histogram to 1 frame.

plt.hist(y_test_full['DispFrames'])
plt.xlabel('Video Frame Rate (in test set) ')
plt.ylabel('Frequencia')
plt.show()

kde = gaussian_kde(y_test_full['DispFrames'])
dist_space = linspace( min(y_test_full['DispFrames']), max(y_test_full['DispFrames']), 100 )
plt.plot( dist_space, kde(dist_space) )
plt.xlabel('Video Frame Rate (in test set) ')
plt.ylabel('Densidade')
plt.show()

# (e) Produce a density plot for the prediction errors y i − ŷ i in the test set.

prediction_errors = np.array(pd.DataFrame(y_test)['DispFrames']) - np.array(pd.DataFrame(y_pred)[0])
kde = gaussian_kde(prediction_errors)
dist_space = linspace( min(prediction_errors), max(prediction_errors), 100 )
plt.plot( dist_space, kde(dist_space) )
plt.xlabel('Predicion Errors')
plt.ylabel('Densidade')
plt.show()

# (f) Based on the above figures and graphs, discuss the accuracy of estimating the Video Frame Rate.

# Observa-se pelo histograma apresentado na letra d que a taxa de frames fica entre 10 e 30, 
# sendo que a maioria da amostra fica de 13 a 15 (mais de 400). O gráfico de densidade da letra E mostra 
# que a taxa de erros varia entre -10 a 10 porém boa parte fica no 0. Entende-se que a taxa de erro
# não é muito alta, sendo que para essa Task pode-se considerar que o modelo foi treinado com um número
# razoável de tentativas.


# In[3]:


# Task II exercício 2

e50 = []
e100 = []
e200 = []
e500 = []
e1000 = []
e2520 = []

# (a) From the above training set with 2520 observations, create six training sets by selecting uniformly
# at random 50, 100, 200, 500, 1000, and 2520 observations (which is the original set).

# (b) Train a linear model and compute the N M AE for each model for the original test set with 1080
# observations.

# (c) Perform the above 50 times, so you train models for 50 different subsets of a given size.

# for criado para gerar os conjuntos de treinamento. Dentro do for é realizada a resposta para as questões
# (a), (b) e (c). Para a questão (a) pode ser considerada a primeira
# iteração do laço. Para a questão (c), após cada predição, os vetores e50, e100, e200, e500, e1000 e e2520 
# são incrementados com o nmae obtido.
for i in range(1, 50):
  X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(x, y, test_size=0.30)
  X_train = X_train_full.drop('TimeStamp', axis = 1)
  y_train = y_train_full.drop('TimeStamp', axis = 1)
  X_test = X_test_full.drop('TimeStamp', axis = 1)
  y_test = y_test_full.drop('TimeStamp', axis = 1)
  X_train_full50, X_test_full50, y_train_full50, y_test_full50 = train_test_split(X_train_full, y_train_full, train_size=50)
  lm = LinearRegression()
  X_train50 = X_train_full50.drop('TimeStamp', axis = 1)
  y_train50 = y_train_full50.drop('TimeStamp', axis = 1)
  X_test50 = X_test_full50.drop('TimeStamp', axis = 1)
  y_test50 = y_test_full50.drop('TimeStamp', axis = 1)
  lm.fit(X_train50, y_train50)
  y_pred = lm.predict(X_test)
  #e50 = e50 + [nmae(y_test, y_pred)]
  e50 = np.append(e50, nmae(y_test, y_pred))
  X_train_full100, X_test_full100, y_train_full100, y_test_full100 = train_test_split(X_train_full, y_train_full, train_size=100)
  lm = LinearRegression()
  X_train100 = X_train_full100.drop('TimeStamp', axis = 1)
  y_train100 = y_train_full100.drop('TimeStamp', axis = 1)
  X_test100 = X_test_full100.drop('TimeStamp', axis = 1)
  y_test100 = y_test_full100.drop('TimeStamp', axis = 1)
  lm.fit(X_train100, y_train100)
  y_pred = lm.predict(X_test)
  #e100 = e100 + [nmae(y_test, y_pred)]
  e100 = np.append(e100, nmae(y_test, y_pred))
  X_train_full200, X_test_full200, y_train_full200, y_test_full200 = train_test_split(X_train_full, y_train_full, train_size=200)
  lm = LinearRegression()
  X_train200 = X_train_full200.drop('TimeStamp', axis = 1)
  y_train200 = y_train_full200.drop('TimeStamp', axis = 1)
  X_test200 = X_test_full200.drop('TimeStamp', axis = 1)
  y_test200 = y_test_full200.drop('TimeStamp', axis = 1)
  lm.fit(X_train200, y_train200)
  y_pred = lm.predict(X_test)
  #e200 = e200 + [nmae(y_test, y_pred)]
  e200 = np.append(e200, nmae(y_test, y_pred))
  X_train_full500, X_test_full500, y_train_full500, y_test_full500 = train_test_split(X_train_full, y_train_full, train_size=500)
  lm = LinearRegression()
  X_train500 = X_train_full500.drop('TimeStamp', axis = 1)
  y_train500 = y_train_full500.drop('TimeStamp', axis = 1)
  X_test500 = X_test_full500.drop('TimeStamp', axis = 1)
  y_test500 = y_test_full500.drop('TimeStamp', axis = 1)
  lm.fit(X_train500, y_train500)
  y_pred = lm.predict(X_test)
  #e500 = e500 + [nmae(y_test, y_pred)]  
  e500 = np.append(e500, nmae(y_test, y_pred))
  X_train_full1000, X_test_full1000, y_train_full1000, y_test_full1000 = train_test_split(X_train_full, y_train_full, train_size=1000)
  lm = LinearRegression()
  X_train1000 = X_train_full1000.drop('TimeStamp', axis = 1)
  y_train1000 = y_train_full1000.drop('TimeStamp', axis = 1)
  X_test1000 = X_test_full1000.drop('TimeStamp', axis = 1)
  y_test1000 = y_test_full1000.drop('TimeStamp', axis = 1)
  lm.fit(X_train1000, y_train1000)
  y_pred = lm.predict(X_test)
  #e1000 = e1000 + [nmae(y_test, y_pred)]
  e1000 = np.append(e1000, nmae(y_test, y_pred))
  lm = LinearRegression()
  X_train2520 = X_train_full.drop('TimeStamp', axis = 1)
  y_train2520 = y_train_full.drop('TimeStamp', axis = 1)
  X_test2520 = X_test_full.drop('TimeStamp', axis = 1)
  y_test2520 = y_test_full.drop('TimeStamp', axis = 1)
  lm.fit(X_train2520, y_train2520)
  y_pred = lm.predict(X_test)
  #e2520 = e2520 + [nmae(y_test, y_pred)]
  e2520 = np.append(e2520, nmae(y_test, y_pred))


print e50
print e100
print e200
print e500
print e1000
print e2520


# In[4]:


# (d) Produce a plot that shows N M AE for M against the size of the training set. Use error bars or
# box plots to show the range of the N M AE values for a given set size

data_to_plot = [e50, e100, e200, e500, e1000, e2520]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data_to_plot)
plt.xlabel('Subsets - 1(50), 2(100), 3(200), 4(500), 5(1000), 6(2520))')
plt.ylabel('NMAE')
plt.show()
