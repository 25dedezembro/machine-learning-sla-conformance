# Task IV exercício 1

# 1. Construct a training set and a test set from the trace as above.

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(x, y, test_size=0.30)
X_train = X_train_full.drop('TimeStamp', axis = 1)
y_train = y_train_full.drop('TimeStamp', axis = 1)
X_test = X_test_full.drop('TimeStamp', axis = 1)
y_test = y_test_full.drop('TimeStamp', axis = 1)

# Task IV exercício 2

# 2. Method 1: Build all subsets of the feature set X that contain either one or two features (i.e., device
# statistics). Compute the models for each of these sets for linear regression over the training set. Plot
# a histogram of the error values (NMAE) of all the models for the test set. Identify the feature set that
# produces the model with the smallest error and give the device statistic(s) in this set.

vector_to_histogram = []
smallest = 1.0
for i in range(0,9):
  lm = LinearRegression() 
  lm.fit(X_train.iloc[:, [i]], y_train)
  y_pred = lm.predict(X_test.iloc[:, [i]])
  error = nmae(y_test, y_pred)
  print("NMAE %s: %.2f" % ( X_test.iloc[:, [i]].keys()[0] , error))
  vector_to_histogram = vector_to_histogram + [error]
  if (error.item() < smallest):
    smallest = error.item()
    smallest_stat = X_train.iloc[:, [i]]
  for j in range(i, 9):
    if (i != j):
      lm = LinearRegression() 
      lm.fit(X_train.iloc[:, [i,j]], y_train)
      y_pred = lm.predict(X_test.iloc[:, [i,j]])
      error = nmae(y_test, y_pred)
      print("NMAE %s, %s: %.2f" % ( X_test.iloc[:, [i,j]].keys()[0], X_test.iloc[:, [i,j]].keys()[1] , error))
      vector_to_histogram = vector_to_histogram + [error]
      if (error.item() < smallest):
        smallest = error.item()
        smallest_stat = X_train.iloc[:, [i,j]]
 
print vector_to_histogram
print smallest
print smallest_stat


# In[11]:


plt.hist(vector_to_histogram)
plt.xlabel('NMAE of all models')
plt.ylabel('Frequencia')
plt.show()


# In[12]:


plt.hist(np.around(vector_to_histogram, decimals=2))
plt.xlabel('Rounded NMAE of all models')
plt.ylabel('Frequencia')
plt.show()


# In[13]:


# Task IV exercício 3

# 3. Method 2: Linear univariate feature selection. Take each feature of X and compute the 
# ...
# Produce a plot that shows the error value in function of the set k.

def sample_correlation(x_obs, y_obs):
  Xs = pd.DataFrame(x_obs)
  Ys = pd.DataFrame(y_obs)
  m = len(Ys)
  E = 0.0
  for i in range(0, m):
    E += (Xs.iloc[i][0] - Xs.mean().item()) * (Ys.iloc[i][0] - Ys.mean().item()) / (Xs.std().item() * Ys.std().item())
  result = 1.0/m * E
  return result

import math

correlations = np.empty((0,3), dtype=float)
for i in range(0,9):
  key = X_train.iloc[:, [i]].keys()[0]
  value = sample_correlation(X_train.iloc[:, [i]], y_train)
  correlations = np.append(correlations, np.array([[i,value,math.pow(value,2)]]), axis=0)
  print('%s: %f square: %f' % (key, value, math.pow(value, 2)))


# In[14]:


print correlations


# In[15]:


rank = correlations[correlations[:, 2].argsort()][::-1]
print rank


# In[97]:


k = []
errors_k = []

for i in range(0,9):
    sTrain = 'X_train.iloc[:, ['
    sTest = 'X_test.iloc[:, ['
    for j in range(0,i+1):
        sTrain += str(int(rank[j][0])) + ','
        sTest += str(int(rank[j][0])) + ','
    sTrain += ']]'
    sTest += ']]'
    temp_train = eval(sTrain)
    temp_test = eval(sTest)
    lm = LinearRegression()
    lm.fit(temp_train, y_train)
    temp_y_pred = lm.predict(temp_test)
    error = nmae(y_test, temp_y_pred)
    k = k + [",".join(temp_train.columns.values.tolist())]
    errors_k = errors_k + [error] 
    
print k
print errors_k


# In[106]:


vector_k = []
for i in range(0,9): 
    vector_k = vector_k + ['K0..' + str(i)]
    
print vector_k
    
plt.xlabel('k')
plt.ylabel('NMAE')
plt.plot(vector_k, errors_k,label='Errors by set k')
legend = plt.legend(loc='best', shadow=True)
plt.show()

