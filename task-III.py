
from sklearn.linear_model import LogisticRegression

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(x, y, test_size=0.30)
logistic = LogisticRegression()

X_train = X_train_full.drop('TimeStamp', axis = 1)
y_train = y_train_full.drop('TimeStamp', axis = 1)
X_test = X_test_full.drop('TimeStamp', axis = 1)
y_test = y_test_full.drop('TimeStamp', axis = 1)

y_train_int = y_train['DispFrames'] >= 18
C = logistic.fit(X_train, y_train_int)

# Task III exercício 1

# 1. Model Training - use Logistic Regression to train a classifier C with the training set. Provide the
# coefficients (Θ 0 , ..., Θ 9 ) of your model C. (Θ 0 is the offset.)
coefficients = C.coef_.tolist()
print coefficients


# In[6]:


y_test_int = y_test['DispFrames'] >= 18

y_pred_int = logistic.predict(X_test)

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test_int, y_pred_int).ravel()

# Task III exercício 2

# 2. Accuracy of the Classifiers C - Compute the classification error (ERR) on the test set for C. For this,
# you first compute the confusion matrix, which includes the four numbers True Positives (TP), True
# Negatives (TN), False Positives (FN), and False Negatives (FN). We define the classification error as
# +T N # , whereby m is the number of observations in the test set. A true positive is an
# ERR = 1 − T P m # observation that is correctly classified by the classifier as conforming to the SLA; 
# a true negative is an observation that is correctly classified by the classifier as violating the SLA.

print (tn, fp, fn, tp)
m = len(y_test_int) * 1.0
ERR = 1 - (tp + tn)/m
print 'ERR: ', ERR


# In[7]:


# Task III exercício 3

# 3. As a baseline for C, use a naı̈ve method which relies on Y values only, as follows. For each x ∈ X,
# the naı̈ve classifier predicts a value T rue with probability p and F alse with probability 1 − p. p is the
# fraction of Y values that conform with the SLA. Compute p over the training set and the classification
# error for the naı̈ve classifier over the test set.

cont = 0
tam = len(y_train_int)
for value in y_train_int:
  if (value == True):
    cont = cont + 1


p = 1.0 * cont / tam
print p
print 1 - p

naive_tam = len(y_test)
naive = np.random.choice([True, False], size=(naive_tam), p=[p, (1-p)])

naive_tn, naive_fp, naive_fn, naive_tp = confusion_matrix(y_test_int, naive).ravel()

print (naive_tn, naive_fp, naive_fn, naive_tp)

naive_m = len(naive) * 1.0
ERR = 1 - (naive_tp + naive_tn)/naive_m
print 'ERR Naive: ', ERR


# In[8]:


# Task III exercício 4

# 4. Build a new classifier by extending extend the linear regression function developed in Task II with a
# check on the output, i.e., the Video Frame Rate. If the frame rate for a given X is above the SLA
# threshold, then the Y label of the classifier is set to conformance, otherwise to violation. Compute the
# new classifier over the training set and the classification error for this new classifier over the test set.

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(x, y, test_size=0.30)
lm = LinearRegression()
X_train = X_train_full.drop('TimeStamp', axis = 1)
y_train = y_train_full.drop('TimeStamp', axis = 1)
X_test = X_test_full.drop('TimeStamp', axis = 1)
y_test = y_test_full.drop('TimeStamp', axis = 1)
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

y_pred_int = y_pred >= 18
y_test_int = y_test['DispFrames'] >= 18
tn, fp, fn, tp = confusion_matrix(y_test_int, y_pred_int).ravel()
print (tn, fp, fn, tp)

y_pred_m = len(y_pred) * 1.0
ERR = 1 - (tp + tn)/y_pred_m
print 'ERR Class: ', ERR
