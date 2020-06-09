# Titanic_practice

import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from sklearn.ensemble import RandomForestClassifier

# 載入資料
train = pd.read_csv('Titanic/titanic_train.csv')
test = pd.read_csv('Titanic/titanic_test.csv')
consequence = pd.read_csv('Titanic/gender_baseline.csv')

print('Shape of train/test/consequence:', train.shape , test.shape , consequence.shape)

test.head(5)
consequence.head(5)
train.head(5)
# 檢視後發現應先合併資料

# 合併：測試資料和結果併為test_merge,再跟訓練資料train合併
test_merge = pd.concat([test, consequence], axis=1, join='inner')
test_merge = test_merge.loc[:,~test_merge.columns.duplicated()]
test_merge.head(5)

result = pd.concat([train,test_merge], keys=['train', 'test'])
result.shape

# 檢查資料缺值
print(result.info())
print('\n')
print(result.isnull().sum())
print('\n')
print(result.describe(include=['O']))

# 【初步檢視p-class資料與生存之關連】
result[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().round(3).sort_values(by='survived', ascending=False)

# 【處理name-1/3】
# 定義函式取得乘客Title
def get_title(g_name):
    title_search = re.search(' ([A-Za-z]+)\.', g_name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

result['Title'] = result['name'].apply(get_title)
result.head(5)

# print(result.describe(include=['O']))
# result.Title.unique()

# 【處理name-2/3】
# 進一步處理特殊Title, 只留下Miss,Mr,Mrs,Master,Rare五種
result['Title'] = result['Title'].replace(['Dr', 'Col', 'Sir',
       'Rev', 'Major', 'Jonkheer', 'Capt', 'Countess', 'Don',
       'Lady', 'Dona'], 'Rare')

result['Title'] = result['Title'].replace('Mlle', 'Miss')
result['Title'] = result['Title'].replace('Ms', 'Miss')
result['Title'] = result['Title'].replace('Mme', 'Mrs')
result.Title.unique()

result[['Title', 'survived']].groupby(['Title'], as_index=False).mean().round(3).sort_values(by='survived', ascending=False)

# 【處理name-3/3】
# 將Title中各組依存活率依序分配好組別(Mrs,Miss,Master,Rare,Mr)五組
result['Title'] = result['Title'].map({'Mrs': 1, 'Miss': 2,'Master': 3, 'Rare': 4, 'Mr': 5}).astype(int)

# 【處理sex】
result['sex'] = result['sex'].map({'male': 1, 'female': 0}).astype(int)

figure = plt.figure(figsize=(15,8))

agemax = result['age'].max()
agebins = range(0, int(agemax)+2, 5)

plt.subplot(221)
result[result['sex'] == 1]['age'].plot.hist(bins=agebins , label = 'Male',  color='#CDC7E5',edgecolor = 'w')
result[result['sex'] == 0]['age'].plot.hist( bins=agebins ,label = 'Female',color='#FCEEB5',edgecolor = 'w')

plt.ylim(top=120)
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.legend()

plt.subplot(223)
result[result['sex'] == 1]['age'].plot.hist(bins=agebins , label = 'Male',  color='#CDC7E5',edgecolor = 'w')
result[(result['sex'] == 1)&(result['survived'] == 1)]['age'].plot.hist(bins=agebins , label = 'Male_survived', 
                                                                          color='#7776bc' ,edgecolor = 'k')
plt.ylim(top=120)
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.legend()

plt.subplot(224)
result[result['sex'] == 0]['age'].plot.hist(bins=agebins , label = 'Female',  color='#FCEEB5',edgecolor = 'w')
result[(result['sex'] == 0)&(result['survived'] == 1)]['age'].plot.hist(bins=agebins , label = 'Female_survived', 
                                                                          color='#ffec51',edgecolor = 'k')
plt.ylim(top=120)
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.legend()


# 【處理age-1/2】
age_avg = result['age'].mean()
age_std = result['age'].std()
age_null_count = result['age'].isnull().sum()


age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

#將隨機產生的整數陣列填入空值陣列
result.loc[np.isnan(result['age']), 'age'] = age_null_random_list

result['age'] = result['age'].astype(int)

# 【處理age-2/2分組】
list = result.loc[result['survived']==1, 'age']
np.percentile(list, (25, 50, 75), interpolation='midpoint')
# 【決定以10歲作為級距分組】
result.loc[ result['age'] <= 10, 'Age'] = 1
result.loc[(result['age'] > 10) & (result['age'] <= 20), 'Age'] = 2
result.loc[(result['age'] > 20) & (result['age'] <= 30), 'Age'] = 3
result.loc[(result['age'] > 30) & (result['age'] <= 40), 'Age'] = 4
result.loc[(result['age'] > 40) & (result['age'] <= 50), 'Age'] = 5
result.loc[(result['age'] > 50) & (result['age'] <= 60), 'Age'] = 6
result.loc[(result['age'] > 60) & (result['age'] <= 70), 'Age'] = 7
result.loc[ result['age'] > 70, 'Age']=8 ;
result['Age'] = result['Age'].astype(int)
result.Age.unique()

# 【新增family = sibsp + parch】
result['family'] = (result['sibsp']+result['parch']).astype(int)

# 【初步檢視ticket-船票編號顯示多組重複 >>> 家族旅遊共用相同船票編號】
result.ticket.describe()

# 【初步檢視fare-分配傾斜嚴重】
figure = plt.figure(figsize=(9,4.8))
result['fare'].plot.hist(bins = 25, color='#bafff0',ec='#a8a8a8')
plt.legend()

# 【處理fare-1/2用中位數填補空值】
result['fare'] = result['fare'].fillna(train['fare'].median())
# 【處理fare-1/2分1~8組】
result.loc[ result['fare'] <= 20, 'Fare'] = 1
result.loc[(result['fare'] > 20) & (result['fare'] <= 40), 'Fare'] = 2
result.loc[(result['fare'] > 40) & (result['fare'] <= 60), 'Fare'] = 3
result.loc[(result['fare'] > 60) & (result['fare'] <= 80), 'Fare'] = 4
result.loc[(result['fare'] > 80) & (result['fare'] <= 100), 'Fare'] = 5
result.loc[(result['fare'] > 100) & (result['fare'] <=120), 'Fare'] = 6
result.loc[(result['fare'] > 120) & (result['fare'] <=140), 'Fare'] = 7
result.loc[ result['fare'] > 140, 'Fare']=8 ;
result['Fare'] = result['Fare'].astype(int)
result.Fare.unique()

# 【初步檢視cabin 複雜且缺值多】
result.cabin.describe()
result.cabin[result.cabin.notnull()]

# 【建立Cabin 代表艙等代號】
result['Cabin'] = result.cabin.astype(str).str[0]
result.Cabin.unique()

# 【初步檢視Cabin和存活率關係-似乎沒太大關連】
result[['Cabin','survived']].groupby(['Cabin'],as_index=False).mean().round(3).sort_values(by='survived', ascending=False)

# 【初步檢視embarked,boat,body和home.dest,都繁多且跟存活率無關聯】>>>僅以embarked代表
result[['embarked','survived']].groupby(['embarked'],as_index=False).mean().round(3).sort_values(by='survived', ascending=False)

result.info()

# split training set the testing set
train = result[:len(train)]
test = result[len(train):]


# Inputs set and labels
X = result.drop(labels=['survived','passenger_id'],axis=1)
Y = result['survived']


# Show Baseline
Base = ['pclass','sex','age','fare']
Base_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
Base_Model.fit(X[Base], Y)
print('Base oob score :%.5f' %(Base_Model.oob_score_)) #得到很高的score0.86937

# submission
X_Submit = test.drop(labels=['passenger_id'],axis=1)

Base_pred = Base_Model.predict(X_Submit[Base])

submit = pd.DataFrame({"PassengerId": test['passenger_id'],
                      "Survived":Base_pred.astype(int)})
submit.to_csv("submit_Base.csv",index=False)




