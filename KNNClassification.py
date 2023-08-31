#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


#import required libraries
from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#obtaining the dataset and putting it in a dataframe
db = pd.read_csv('C:/Users/Georgina/Desktop/diabetes.csv')

#View the first few rows.
db.head()


# In[4]:


#farmiliarizing yourself with the data youll be dealing with
print(db.shape)


# In[5]:


#descriptive statistics to summarize the central tendency, dispersion and shape of a dataset’s distribution
db.describe()


# In[6]:


#information about the data types & columns
db.info(verbose=True)


# In[7]:


#replace unnecessary 0 values with NaN to make it easier for processing
dbr = db.copy(deep = True)
dbr[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dbr[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

## showing the count of NaN's
print(dbr.isnull().sum())


# In[8]:


#distribution of data
p = db.hist(figsize = (15,25))


# In[9]:


#replace NaN values with mean or median of the column
dbr['Glucose'].fillna(dbr['Glucose'].mean(), inplace = True)
dbr['BloodPressure'].fillna(dbr['BloodPressure'].mean(), inplace = True)
dbr['SkinThickness'].fillna(dbr['SkinThickness'].median(), inplace = True)
dbr['Insulin'].fillna(dbr['Insulin'].median(), inplace = True)
dbr['BMI'].fillna(dbr['BMI'].median(), inplace = True)


# In[10]:


#distribution should be roughly similar
p = dbr.hist(figsize = (15,25))


# In[11]:


# checking the balance of the data by plotting the count of outcomes by their value
#0- dont have diabetes
#1- diabetes patients
color_wheel = {1: "#0392cf", 
               2: "#7bc043"}
colors = db["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(db.Outcome.value_counts())
p=db.Outcome.value_counts().plot(kind="bar")


# In[12]:


#scatter matrix of cleaned data
from pandas.plotting import scatter_matrix
p=scatter_matrix(dbr,figsize=(15, 20))


# In[13]:


#correlation of cleaned data using spearman method
dbr.corr(method = 'spearman')


# In[14]:


#visualize correlation matrix using heatmap
plt.figure(figsize=(12,10))
correlation_matrix = dbr.corr(method = 'spearman')
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix HeatMap')
plt.show()
#darker colors represent a lower correlation


# In[15]:


#scaling the data
#mean of 0
#standard deviation of 1
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x =  pd.DataFrame(sc_x.fit_transform(dbr.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
x.head()


# In[16]:


y = dbr.Outcome


# In[17]:


#splitting data to train set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=42, stratify=y)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(x_train,y_train)
    
    train_scores.append(knn.score(x_train,y_train))
    test_scores.append(knn.score(x_test,y_test))


# In[19]:


## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[20]:


## score that comes from testing on the datapoints that were split in the beginning to be used for testing
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[21]:


#visualizing results
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')


# In[22]:


#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
#True & false positives,true and false negatives
y_pred = knn.predict(x_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[23]:


#classification report which includes Precision, Recall and F-1 score
#Precision: the ratio of correctly predicted positive observations to the total predicted positive observations
#Precision Score - tells us the accuracy of positive predictions
#Recall:the ratio of correctly predicted positive observations to the all observations
#Recall Score :Fraction of positives that were identified
#F-1 Score :compares precision score and recall; the weighted average of Precision and Recall


# In[24]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[25]:


#ROC Curve - Can the model differentiate between diabetes patients and non diabetes patients
from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()


# In[26]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)


# In[27]:


#Hyperparemeter optimization
#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#define range of values
param_grid = {'n_neighbors':np.arange(1,100)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(x,y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))

