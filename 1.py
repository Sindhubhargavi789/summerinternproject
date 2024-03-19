#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('insurance.csv')
data


# In[2]:


copydata=data.copy()
copydata


# In[3]:


copydata.info()


# In[4]:


copydata.describe()


# In[5]:


copydata.dtypes


# In[6]:


copydata.columns


# In[7]:


copydata.shape


# In[8]:


copydata.nunique()


# In[9]:


copydata['sex']=copydata['sex'].map({'male':0,'female':1})
copydata['smoker']=copydata['smoker'].map({'yes':1,'no':0})
copydata['region']=copydata['region'].map({'northwest':0, 'northeast':1,'southeast':2,'southwest':3})
copydata


# In[10]:


copydata.dtypes


# In[11]:


copydata.describe()


# In[12]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x=data['sex'])


# In[13]:


data['sex'].value_counts()


# In[14]:


sns.countplot(x=data['children'])


# In[15]:


sns.countplot(x=data['bmi'])


# In[16]:


sns.countplot(x=data['region'])


# In[17]:


sns.countplot(x=data['age'])


# In[18]:


sns.countplot(x=data['smoker'])


# In[19]:


#bivaraite analysis between output charges and input parameters
features = ['sex', 'children', 'smoker', 'region']
 
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i+1)
    copydata.groupby(col).mean()['charges'].plot.bar()
plt.show()


# In[20]:


sns.scatterplot(x=data['age'],y=data['charges'],hue=data['smoker'])


# In[21]:


sns.scatterplot(x=data['bmi'],y=data['charges'],hue=data['smoker'])


# In[22]:


sns.boxplot(copydata['age'])


# In[23]:


sns.boxplot(copydata['bmi'])


# In[24]:


percentile25=copydata['bmi'].quantile(0.25)
percentile75=copydata['bmi'].quantile(0.75)


# In[25]:


percentile25


# In[26]:


percentile75


# In[27]:


iqr=percentile75-percentile25
iqr


# In[28]:


lower=percentile25-1.5*iqr
lower


# In[29]:


higher=percentile75+iqr*1.5
higher


# In[30]:


copydata[copydata['bmi']<lower]


# In[31]:


copydata[copydata['bmi']>higher]


# In[32]:


newcopydata=copydata[copydata['bmi']<higher]
newcopydata


# In[33]:


newcopydata.shape


# In[34]:


import seaborn as sns
sns.boxplot(newcopydata['bmi'])


# In[35]:


#check the skewness of the data
newcopydata['bmi'].skew()


# In[36]:


newcopydata['age'].skew()


# In[37]:


newcopydata


# In[38]:


newcopydata.corr()


# In[39]:


sns.heatmap(data=newcopydata.corr(),annot=True)


# In[40]:


#linear regression model 
from sklearn.model_selection import train_test_split,cross_val_score
x=newcopydata.drop(['charges'],axis=1)
y=newcopydata[['charges']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[41]:


x_train


# In[42]:


x_test


# In[43]:


y_train


# In[44]:


y_test


# In[45]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[46]:


y_pred=reg.predict(x_test)


# In[47]:


y_pred


# In[48]:


print(np.sqrt(mean_squared_error(y_test,y_pred)))
print(reg.score(x_train,y_train))
print(reg.score(x_test,y_test))
cvscore=cross_val_score(reg,x,y,cv=5)
print(cvscore.mean())


# In[49]:


from pickle import dump
dump(reg,open('insurancemodelf.pkl','wb'))


# In[50]:


new_data=pd.DataFrame({'age':19,'sex':'female','bmi':27.9,'children':0,'smoker':'yes','region':'southwest'},index=[0])
new_data['smoker']=new_data['smoker'].map({'yes':1,'no':0})
new_data['sex']=new_data['sex'].map({'male':0,'female':1})
new_data['region']=new_data['region'].map({'northwest':0,'northeast':1,'southwest':2,'southeast':3})

reg.predict(new_data)


# In[51]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,r2_score
X = newcopydata.drop(['charges'], axis=1)  
Y= newcopydata['charges']


# In[52]:


from sklearn.ensemble import RandomForestRegressor
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[53]:


xtest


# In[54]:


ytest


# In[55]:


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)


# In[56]:


rf_regressor.fit(xtrain, ytrain)


# In[57]:


y_pred = rf_regressor.predict(xtest)
y_pred


# In[58]:


#print(r2_score(xtrain,ytrain))
print(rf_regressor.score(xtest,ytest))
print(rf_regressor.score(xtrain,ytrain))
print(cross_val_score(rf_regressor,X,Y,cv=5,).mean())


# In[59]:


from pickle import dump
dump(rf_regressor,open('insurancemodelf.pkl','wb'))


# In[60]:


new_data1=pd.DataFrame({'age':19,'sex':'female','bmi':27.9,'children':0,'smoker':'yes','region':'southwest'},index=[0])
new_data1['smoker']=new_data1['smoker'].map({'yes':1,'no':0})
new_data1['sex']=new_data1['sex'].map({'male':0,'female':1})
new_data1['region']=new_data1['region'].map({'northwest':0,'northeast':1,'southwest':2,'southeast':3})

rf_regressor.predict(new_data1)


# In[61]:


from sklearn.ensemble import GradientBoostingRegressor
X = newcopydata.drop(['charges'], axis=1)  
Y= newcopydata['charges']

x1train,x1test,y1train,y1test=train_test_split(X,Y,test_size=0.2,random_state=i)


# In[62]:


x1train


# In[63]:


y1train


# In[64]:


x1test


# In[65]:


y1test


# In[66]:


grad = GradientBoostingRegressor(
    loss='squared_error',  
    learning_rate=0.2,
    n_estimators=15,
    max_depth=5,
    random_state=42
)
grad


# In[67]:


grad.fit(x1train,y1train)
res=grad.predict(x1test)
res


# In[68]:


print(grad.score(x1test,y1test))
print(grad.score(x1train,y1train))
print(cross_val_score(grad,X,Y,cv=5).mean())


# In[69]:


from pickle import dump
dump(grad,open('insurancemodel.pkl','wb'))


# In[70]:


new_data1=pd.DataFrame({'age':19,'sex':'female','bmi':27.9,'children':0,'smoker':'yes','region':'southwest'},index=[0])
new_data1['smoker']=new_data1['smoker'].map({'yes':1,'no':0})
new_data1['sex']=new_data1['sex'].map({'male':0,'female':1})
new_data1['region']=new_data1['region'].map({'northwest':0,'northeast':1,'southwest':2,'southeast':3})

grad.predict(new_data1)


# In[79]:


# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('insurancemodel.pkl')

# Define the Streamlit app
def main():
    st.title('Health Insurance Prediction System')

    # Create input fields
    age = st.slider('Select Age', 18, 100, 25)
    sex = st.selectbox('Select Gender', ('male', 'female'))
    bmi = st.number_input('Enter BMI', min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    smoker = st.selectbox('Are you a smoker?', ('no', 'yes'))
    region = st.selectbox('Select Region', ('northwest', 'southwest', 'northeast', 'southeast'))

    # Convert categorical values to numerical
    sex = 0 if sex == 'male' else 0
    smoker = 1 if smoker == 'yes' else 0

    # Encode region variable
    region_mapping = {'northwest': 0, 'southwest': 1, 'northeast': 2, 'southeast': 3}
    region_encoded = region_mapping[region]

    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({'age': [age], 'sex': [sex],'bmi': [bmi], 'children': [children],  'smoker': [smoker], 'region': [region_encoded]})

    # Make prediction
    if st.button('Predict'):
        prediction = model.predict(input_data)
        st.success(f'The predicted insurance status is: {prediction[0]}')

# Run the app
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




