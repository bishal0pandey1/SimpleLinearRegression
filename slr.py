
#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING DATASET AND DIVIDING INTO DEPENDENT AND INDEPENDENT VARIBALES
df=pd.read_csv(r"C:\Users\desti\Desktop\DATASCIENCE\ML\REGRESSION\SimpleLinearRegression\Salary_Data.csv")

#DATA PRE-PROCESSING AND CLEANSING
#FIRST FIVE ROWS
df.head()
#NO OF ROWS AND COLUMNS
df.shape

#COLUMNS NAME
df.columns

#INFORMATION ABOUT DATAFRAME IF THERE ARE MISSING VALUES
df.info()
df.isnull().sum()

#Descriptive statistics 
df.describe()

#SEPERATING DATA INTO DEPENDENT AND INDEPENDENT VARIABLES
x=df.iloc[ : , :-1].values
y=df.iloc[ : , 1].values

#VISUALIZATION

plt.hist(y,label='Salary')
plt.title("Histogram Plot ")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.scatter(x,y,color='red',label='Original data')    #shows positive co-relation
plt.xlabel("Years of Experience")
plt.plot(x,y,color='black')
plt.ylabel("Salary")
plt.title('Years of Experience Vs Salary')
plt.legend()
plt.show()


#SPLITTING DATA INTO TRAINING AND TESTING PHASE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#MODEL SELECTION
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

#MODEL IS TRAINED NOW ,LETS PREDICT
y_predict=model.predict(x_test)


#COMPARISION OF TRAINING ,TEST AND PREDICTED DATA 
plt.scatter(x_train, y_train, color='red', label='Training Data')
plt.scatter(x_test, y_test, color='yellow', label='Test Data')
plt.scatter(x_test,y_predict,color='black',label='Predicted Data')
plt.plot(x_test, y_predict, color='blue', label='Best Fit Line')
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

#MAKING NEW PREDICTIONS

pred_salary=np.array([8,10.5,8.2]).reshape(-1,1)   #to convert it into column
pred=model.predict(pred_salary)
print(pred)

#CHECKING ACCURACY
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2 is ",r2)

import statsmodels.api as sm
regressor_OLS=sm.OLS(endog=y,exog=x).fit()
regressor_OLS.summary()








