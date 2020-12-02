#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy
from sklearn.preprocessing import LabelEncoder
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.utils import resample
df1 = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",names=['buying','maint','door','persons','lug_boot','safety','review'])
df1.fillna(0)
print(df1)

array = df1.values
# separate array into input and output components
X, Y = array[:, 0:7], array[:, 4]
# ensure inputs are floats and output is an integer label
Le= LabelEncoder()
X[:, 0] = Le.fit_transform(X[:, 0])
X[:, 1] = Le.fit_transform(X[:, 1])
X[:, 4] = Le.fit_transform(X[:, 4])
X[:, 2] = Le.fit_transform(X[:, 2])
X[:, 3] = Le.fit_transform(X[:, 3])
X[:, 5] = Le.fit_transform(X[:, 5])
X[:, 6] = Le.fit_transform(X[:, 6])

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:4,:])
print (rescaledX.mean(axis = 0))      # tính giá trị trung bình mỗi cột
print (rescaledX.std(axis = 0))       # tính giá trị phương sai mỗi cột


# In[2]:


good, vgood, unacc, acc = df1.review.value_counts()
print(f'Good: {good}')
print(f'Vgood: {vgood}')
print(f'unacc: {unacc}')
print(f'acc: {acc}')
sns.countplot(x = 'review', data = df1);


# In[3]:


df_2 = df1[df1.review == 2]
df_0 = df1[df1.review == 0]
df_3 = df1[df1.review == 3]
df_1 = df1[df1.review == 1]
count_class_0, count_class_1, count_class_2, count_class_3 = df1.review.value_counts()
# # print(count_class_0)
# # Oversampling
random_seed = 5180440
np.random.seed(random_seed)
df_new_1 = resample(df_1, replace=True, n_samples=len(df_2), random_state=random_seed)
df_new_0 = resample(df_0, replace=True, n_samples=len(df_2), random_state=random_seed)
df_new_3 = resample(df_3, replace=True, n_samples=len(df_2), random_state=random_seed)
df_test_under = pd.concat([df_new_1, df_new_0, df_new_3, df_2])
print(df_test_under.review.value_counts())
sns.countplot(x = 'review', data=df_test_under);


# In[4]:


X=df1.iloc[:,:-1]
Y = df1.loc[:,["review"]]
le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
X=X.apply(le1.fit_transform)
Y=Y.apply(le2.fit_transform)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)


# In[5]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
y_pred = neigh.fit(X_train, y_train.values.ravel())
predictions = neigh.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[6]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train.values.ravel())
predictions = gnb.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# In[7]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train.values.ravel())
predictions1 = gnb.predict(X_test)
print(confusion_matrix(y_test,predictions1))
print(classification_report(y_test,predictions1))


# In[8]:


from sklearn.naive_bayes import BernoulliNB
clf1 = BernoulliNB()
clf1.fit(X_train, y_train.values.ravel())
predictions2 = gnb.predict(X_test)
print(confusion_matrix(y_test,predictions2))
print(classification_report(y_test,predictions2))


# In[9]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf2.fit(X_train, y_train.values.ravel())
predictions2 = clf2.predict(X_test)
print(confusion_matrix(y_test,predictions2))
print(classification_report(y_test,predictions2))


# In[10]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
clf3 = MLPClassifier(hidden_layer_sizes=(10,10,10),random_state=1, max_iter=300).fit(X_train, y_train.values.ravel())
predictions2 = clf3.predict(X_test)
print(confusion_matrix(y_test,predictions2))
print(classification_report(y_test,predictions2))


# In[12]:


import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='sgd', loss='mse',metrics=['accuracy'])
# This builds the model for the first time:
clf4=model.fit(X_train, y_train, batch_size=32, epochs=10)


# In[13]:


es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4000, verbose=0, callbacks=[es])
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




