import numpy as np
import pandas as pd
import pickle

data=pd.read_csv('proiths1(linreg).csv')


X=data['x(A*300sqft)'].values
Y=data['y(C*10 lakh rs)'].values

X=X.reshape(5,1)
Y=Y.reshape(5,1)

"""def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['x(A*300sqft)'] = X['x(A*300sqft)'].apply(lambda x : convert_to_int(x))"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts

X_train,X_test,Y_train,Y_test=tts(X,Y,train_size=0.6,test_size=0.4,random_state=4)
reg=LinearRegression()
reg.fit(X_train,Y_train)

#Saving the model

pickle.dump(reg,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[4]]))