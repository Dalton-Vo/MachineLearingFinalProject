#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from sklearn import metrics, neighbors, datasets, svm
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df1 = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",names=['buying','maint','door','persons','lug_boot','safety','review'])
df1.head()


# In[8]:


X=df1.iloc[:,:-1]
Y = df1.loc[:,["review"]]
le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
X=X.apply(le1.fit_transform)
Y=Y.apply(le2.fit_transform)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
y_pred = neigh.fit(X_train, y_train.values.ravel())
predictions = neigh.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[12]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train.values.ravel())
predictions = gnb.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))



# In[13]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train.values.ravel())
predictions1 = gnb.predict(X_test)
print(confusion_matrix(y_test,predictions1))
print(classification_report(y_test,predictions1))


# In[14]:


from sklearn.naive_bayes import BernoulliNB
clf1 = BernoulliNB()
clf1.fit(X_train, y_train.values.ravel())
predictions2 = gnb.predict(X_test)
print(confusion_matrix(y_test,predictions2))
print(classification_report(y_test,predictions2))


# In[16]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf2.fit(X_train, y_train.values.ravel())
predictions2 = clf2.predict(X_test)
print(confusion_matrix(y_test,predictions2))
print(classification_report(y_test,predictions2))


# In[18]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
clf3 = MLPClassifier(hidden_layer_sizes=(10,10,10),random_state=1, max_iter=300).fit(X_train, y_train)
predictions2 = clf3.predict(X_test)
print(confusion_matrix(y_test,predictions2))
print(classification_report(y_test,predictions2))


# In[2]:


from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4000, verbose=0, callbacks=[es])
# evaluate the model
train_acc = model.evaluate(X_train, y_train, verbose=0)
test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()


# In[ ]:




