import pandas as pd
import pickle as pkl
hiringDf=pd.read_csv('hiring.csv')
print(hiringDf.head())
y=hiringDf['salary']
X=hiringDf.drop(['salary'],axis=1)
from sklearn.linear_model  import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
pkl.dump(regressor, open('model.pkl','wb'))
model=pkl.load(open('model.pkl','rb'))
print(model.predict([[2,8,8]]))

