#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.preprocessing import StandardScaler, PolynomialFeatures	# StandardScaler 将特征正则化；PolynomialFeatures 用于生成多项式的、互动的特征数据
from sklearn.linear_model import LogisticRegression
from  sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import  matplotlib as mpl
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR


# In[2]:


path = 'housing - data-after.csv'
data = pd.read_csv(path,sep=',')

data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
data.info()
data.head()
data.describe()


# In[3]:


X ,y = data[data.columns.delete(-1)], data['MEDV']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=888)
#划分训练集、测试集


# In[4]:


#pipeline = make_pipeline(X_train, y_train)
#scores = cross_val_score(pipeline,X=X_train, y=y_train, cv=10,n_jobs=1)

kf = KFold(n_splits=10, random_state=None) # 10折
#10折划分训练集测试集
for train_index, test_index in kf.split(X):
      
      X_train, X_test = X.loc[train_index], X.loc[test_index] 
      y_train, y_test = y.loc[train_index], y.loc[test_index] 
      

      #线性回归模型
      linear_model = LinearRegression(normalize=True,n_jobs=-1)
      linear_model.fit(X_train, y_train)
    
      # 岭回归模型
      Ridge_model = RidgeCV(alphas=1,normalize=True)
      Ridge_model.fit(X_train, y_train)
    
#线性回归模型
Lcoef = linear_model.coef_#回归系数
line_pre = linear_model.predict(X_test)
print('Linear:')
print('SCORE:{:.4f}'.format(linear_model.score(X_test, y_test)))
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, line_pre))))
print('coef:',Lcoef)

#岭回归模型
Rcoef = Ridge_model.coef_#回归系数
Ridge_pre = Ridge_model.predict(X_test)
print('Ridge:')
print('SCORE:{:.4f}'.format(Ridge_model.score(X_test, y_test)))
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, Ridge_pre))))
print('coef:',Rcoef)


#拟合情况
hos_pre = pd.DataFrame()
hos_pre['line_Predict'] = line_pre
hos_pre['Ridge_Predict'] = Ridge_pre
hos_pre['Truth'] = y_test.reset_index(drop=True)
hos_pre.plot(figsize=(18,8))

#linear模型、Ridge模型相关系数
df_coef = pd.DataFrame()
df_coef['Title'] = data.columns.delete(-1)
df_coef['RCoef'] = Rcoef
df_coef['LCoef'] = Lcoef
df_coef.plot()


#model = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)
#model.fit(X,y)


# In[ ]:





# In[9]:


z = np.array(data['RM']).reshape(-1,1)
w = np.array(data['MEDV']).reshape(-1,1)
print(z.shape)

KF=KFold(n_splits=10, shuffle=True)
poly = PolynomialFeatures(degree = 4)
poly.fit(z)
z_poly = poly.transform(z)
loss = 0
for train_index,test_index in KF.split(z_poly):
    z_found,z_exam=z_poly[train_index],z_poly[test_index]
    w_found,w_exam=w[train_index],w[test_index]
    li = LinearRegression(normalize=True,n_jobs=-1)
    li.fit(z_found,w_found)
    w_pred = li.predict(z_exam)
    lg = RidgeCV(normalize=True)
    lg.fit(z_found, w_found)
    w_train = lg.predict(z_exam)
    loss = loss + mean_squared_error(w_exam, w_pred)

print(loss)

t = np.arange(len(X_test))
s = np.arange(len(z_exam))
plt.plot(s, w_exam, 'r',label='line_Predict')
plt.plot(s, w_pred, 'b',label='Ridge_Predict')
plt.plot(s, w_train, 'g',label='Truth')

# 设置图例
plt.legend(loc='upper right')
plt.show()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




