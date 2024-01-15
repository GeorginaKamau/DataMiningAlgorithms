#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-learn xgboost


# In[2]:


#Gender is 1 if a respondent is male and 0 if a respondent is female.
#Age is a respondent’s age in years.
#family_history_with_overweight is 1 if a respondent has family member who is or was overweight, 0 if not.
#FAVC is 1 if a respondent eats high caloric food frequently, 0 if not.
#FCVC is 1 if a respondent usually eats vegetables in their meals, 0 if not.
#NCP represents how many main meals a respondent has daily (0 for 1-2 meals, 1 for 3 meals, and 2 for more than 3 meals).
#CAEC represents how much food a respondent eats between meals on a scale of 0 to 3.
#SMOKE is 1 if a respondent smokes, 0 if not.
#CH2O represents how much water a respondent drinks on a scale of 0 to 2.
#SCC is 1 if a respondent monitors their caloric intake, 0 if not.
#FAF represents how much physical activity a respondent does on a scale of 0 to 3.
#TUE represents how much time a respondent spends looking at devices with screens on a scale of 0 to 2.
#CALC represents how often a respondent drinks alcohol on a scale of 0 to 3.
#Automobile, Bike, Motorbike, Public_Transportation, and Walking indicate a respondent’s primary mode of transportation. Their primary mode of transportation is indicated by a 1 and the other columns will contain a 0.
#NObeyesdad is a 1 if a patient is obese and a 0 if not.


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[4]:


data = pd.read_csv("obesity.csv")


# In[5]:


#view first few rows of data
data.head()


# In[6]:


#getting info about the data-- datatypes
data.info()


# In[7]:


data.shape


# In[8]:


#19 columns and 2111 rows


# In[9]:


data.isna().sum()


# In[10]:


#column analysis - categorical & numerical variables
def grab_col_names(dataframe, cat_th=10):
    #cat_cols
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes =="0"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
    			dataframe[col].dtypes != "0"]
    cat_cols = cat_cols + num_but_cat
   #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes !="0"]
    num_cols = [col for col in num_cols if col not in cat_cols]
    
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Number of Categorical Variables: {len(cat_cols)}")
    print(f"Number of Numerical Variables: {len(num_cols)}")
        
    return cat_cols, num_cols
cat_cols, num_cols = grab_col_names(data)


# In[11]:


categorical = print(cat_cols)
Numerical =print(num_cols)


# In[12]:


#It is important to understand the dataset beforehand, the only numerical variable is age but since some of the columns are a scale pof values form 0 to 3, they will be classified as numerical


# In[13]:


#unique value analysis - how many different values e.g(there's 2 entries for gender[1 & 0])
dict = {}
for i in list(data.columns):
    dict[i] = data[i].value_counts().shape[0]
pd.DataFrame(dict, index = ["unique count"]).transpose()


# In[14]:


#since most of the columns are categorical, 1 being yes and 0 being no
#We'll perform descriptive statistics for the age column only
column_name = 'Age'
column_statistics = data[column_name].describe()
#print output
print(column_statistics)


# In[15]:


#numerical variables distribution
def histPlot(num):
    sns.histplot(data = data, x = num, bins = 50, kde = True)
    print("{} distribution with hist:".format(num))
    plt.show()
for i in num_cols:
    histPlot(i)


# In[16]:


#Number of ov=bese people
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x=data["NObeyesdad"], palette="magma")
plt.show()


# In[17]:


#male vs female
# Gender

fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x=data["Gender"], palette="tab10")
plt.show()


# In[18]:


# Select relevant columns for correlation analysis
selected_columns = ['Age', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking', 'NObeyesdad']
selected_data = data[selected_columns]

# Calculate the correlation matrix
correlation_matrix = selected_data.corr()

# Display the correlation matrix
print(correlation_matrix)


# In[19]:


##+ve correlation--as one variable increases, the other one increases proportionally,linear correlation  +ve slope, +ve numbers
## -ve correlation---as one variable increases, the other one reduces proportionally,linear correlation-- -ve slope, -ve numbers


# In[20]:


#correlation analysis using heatmap-data vizualization
plt.figure(figsize = (14,10))
sns.heatmap(data.corr(), annot = True, fmt = ".1f", linewidths = .7, cmap="Blues")
plt.show()


# In[21]:


#'NObeyesdad' is the target variable
X = data.drop('NObeyesdad', axis=1)  # Features
y = data['NObeyesdad']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


#which algorithm would work best 
dtc = DecisionTreeClassifier(criterion="entropy")
rfc = RandomForestClassifier()
gnb = GaussianNB()
xgb = XGBClassifier(learning_rate=0.9)

models = [dtc, rfc, gnb, xgb]
names = ["Decision Tree", "Random Forest", "Gaussian Naive Bayes", "XGB"]


# In[23]:


# Train and test each model on the split data
for model, name in zip(models, names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)              


# In[24]:


#print the precision, recall, and F1 score for each class 
# Print classification report and confusion matrix
print(f"{name}:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[25]:


#Remove the column NObeyesdad from the dataframe; seperate into features used for predicting (x) and y 
X = data.drop('NObeyesdad', axis=1)  # Features
y = data['NObeyesdad']  # Target variable


# In[26]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


#load the XGB model 
xgb_model = XGBClassifier()


# In[28]:


# Define a parameter grid for hyperparameter tuning
param_dist = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
}


# In[29]:


# Define the RandomizedSearchCV
random_search = RandomizedSearchCV(
    xgb_model, param_distributions=param_dist, scoring='accuracy', cv=5, n_iter=10, random_state=42
)


# In[30]:


# Perform the randomized search for hyperparameter tuning
random_search.fit(X_train, y_train)


# In[31]:


# Get the best parameters from the search
best_params = random_search.best_params_


# In[32]:


# Train the XGBoost model with early stopping using the best parameters
xgb_model_best = XGBClassifier(
    learning_rate=best_params['learning_rate'],
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_child_weight=best_params['min_child_weight'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
)


# In[33]:


# Enable early stopping
eval_set = [(X_test, y_test)]
xgb_model_best.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)


# In[34]:


# Make predictions on the test set
y_pred = xgb_model_best.predict(X_test)


# In[35]:


# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Best XGBoost Model Accuracy:", accuracy)


# In[37]:


#'selected_features' is a list of features used for prediction
selected_features = ['Age', 'Gender', 'FAF', 'FCVC', 'CAEC', 'CH2O', 'SCC', 'family_history_with_overweight']

# Accept user inputs for the selected variables
# Accept user inputs for the selected variables
user_inputs = {}
for feature in selected_features[:-1]:  # Exclude the target variable 'NObeyesdad'
    user_input = input(f"Enter value for {feature}: ")
    user_inputs[feature] = [float(user_input)]


# In[38]:


# Create a DataFrame from user inputs
user_data = pd.DataFrame(user_inputs)


# In[39]:


# Load the trained XGBoost model
xgb_model = XGBClassifier(learning_rate=0.9)


# In[43]:


#'X_train' and 'y_train' are your training data
X_train = data[['Age', 'Gender', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']]
y_train = data['NObeyesdad']
xgb_model.fit(X_train, y_train)


# In[48]:


# Separate features (X) from user data
user_features = user_data
user_features = user_features.reindex(columns=X_train.columns, fill_value=0)


# In[49]:


# Make predictions for the user input
user_prediction_proba = xgb_model.predict_proba(user_features)[:, 1]
user_prediction = (user_prediction_proba > 0.5).astype(int)


# In[50]:


# Display the predicted probability and binary prediction
print("Predicted Probability:", user_prediction_proba[0])
print("Binary Prediction (1 for obese, 0 for not obese):", user_prediction[0])


# In[ ]:




