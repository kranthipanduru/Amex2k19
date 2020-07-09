#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("TrainingData.csv")
lb_data=pd.read_csv("testX.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.count()


# In[6]:


from sklearn.model_selection import GridSearchCV,StratifiedKFold, train_test_split
kfold = StratifiedKFold(n_splits=5)
from sklearn.metrics import roc_auc_score,roc_curve,auc
from imblearn.over_sampling import SMOTE


# In[7]:


from sklearn import preprocessing


# In[8]:


label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df.mvar47)
df.mvar47=label_encoder.transform(df.mvar47)
lb_data.mvar47=label_encoder.transform(lb_data.mvar47)


# In[9]:


for col in df.columns.values.tolist():
    if df[col].dtypes == 'object':
        df[col].replace(['na','missing'],[np.nan,np.nan], inplace=True)
        df[col]=df[col].astype("float")   

for col in lb_data.columns.values.tolist():
    if lb_data[col].dtypes=='object':
        lb_data[col].replace(['na','missing'],[np.nan,np.nan], inplace=True)
        lb_data[col]=lb_data[col].astype("float")


# In[10]:


from scipy.stats import skew
from scipy.special import boxcox1p


# In[11]:


numeric_features = df.dtypes[df.dtypes != "object"].index


# In[12]:


numeric_features = df[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :numeric_features})
highly_skewed = skewness.index.values[:20]


# In[13]:


lam = 0.15
for feat in highly_skewed:
    df[feat] = boxcox1p(df[feat], lam)
    lb_data[feat] = boxcox1p(lb_data[feat], lam)


# In[14]:


df['Total_Severity']=df['mvar3']+df['mvar4']+df['mvar5']

df['max_credit_avail']=df['mvar7']+df['mvar8']

df['total_num_avail']= df['mvar17']+df['mvar18']

df['total_75%_credit']= df['mvar19']+df['mvar20']


# In[15]:


lb_data['Total_Severity']=lb_data['mvar3']+lb_data['mvar4']+lb_data['mvar5']

lb_data['max_credit_avail']=lb_data['mvar7']+lb_data['mvar8']

lb_data['total_num_avail']= lb_data['mvar17']+lb_data['mvar18']

lb_data['total_75%_credit']= lb_data['mvar19']+lb_data['mvar20']


# In[16]:


Y_train = df["default_ind"]
X_train = df.drop(["default_ind","application_key"], axis=1)
X_train.shape, Y_train.shape


# In[17]:


train_x,val_x,train_y,val_y = train_test_split(X_train, Y_train, test_size = 0.20, random_state=14)
train_x.shape,val_x.shape,train_y.shape,val_y.shape


# In[18]:


import lightgbm
train_data = lightgbm.Dataset(train_x, label=train_y)
test_data = lightgbm.Dataset(val_x, label=val_y)


# In[19]:


para={'boosting_type': 'gbdt',
 'colsample_bytree': 0.65,
 'learning_rate': 0.005,
 'max_bin': 512,
 'max_depth': -1,
 'metric': 'auc',
 'min_child_samples': 5,
 'min_child_weight': 1,
 'min_split_gain': 0.5,
 'nthread': 3,
 'num_class': 1,
 'num_leaves': 23,
 'objective': 'binary',
 'reg_alpha': 0.8,
 'reg_lambda': 1.2,
 'scale_pos_weight': 1,
 'subsample': 0.7,
 'subsample_for_bin': 200,
 'subsample_freq': 1}



model = lightgbm.train(para,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=50,
                      )


# In[20]:


app_key1=lb_data.application_key
lb_data=lb_data.drop("application_key",axis=1)
x = lb_data.values
y_pred1 = model.predict(x)
a = np.where(y_pred1 > 0.5, 1, 0)


# In[22]:


print("Importance of features...")
gain = model.feature_importance('gain')
ft = pd.DataFrame({'feature':model.feature_name(), 'split':model.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft.head(10))


# In[23]:


y_test_pred = model.predict(val_x)
a_test_predict = np.where(y_test_pred > 0.5, 1, 0)
from sklearn.metrics import classification_report
print(classification_report(val_y, a_test_predict))


# In[24]:


from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(val_y, a_test_predict)


# In[25]:


from sklearn.metrics import f1_score
f1_score(val_y,a_test_predict)


# In[26]:


from sklearn.metrics import accuracy_score
accuracy_score(val_y, a_test_predict)


# In[33]:


def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))

y_keys = list(val_y.keys())
y_val = []
for i in y_keys:
    y_val.append(val_y[i])

print("Gini Index =",gini(y_val.count(0)/len(y_val)))


# In[27]:


submission1 = pd.DataFrame({
        "application_key": app_key1,
        "prob": y_pred1,
        "default_ind": a
    })
submission1.default_ind.value_counts()


# In[28]:


submission1 = pd.DataFrame({
        "application_key": app_key1,
        "prob": y_pred1,
        "default_ind": a
    })
submission1.default_ind.value_counts()


# In[29]:


submission1.head()


# In[30]:


submsn = submission1.drop("prob",axis = 1)
submsn.to_csv('without_sorting.csv',header=False, index = False)


# In[31]:


submission1 = submission1.sort_values(["prob"], ascending = 1)
submission1.dtypes


# In[32]:


submission1=submission1.drop("prob",axis=1)
submission1.to_csv('Group_0001.csv',header=False, index=False)

