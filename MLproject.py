import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error , r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import warnings

df = pd.read_excel('students.xlsx')

print(df)

print(df.head())

#checking for null values
print(df.isnull().sum())

#checking for duplicates
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)

X = df.drop(columns=['math score'])
print(X)
print(X.head())
Y = df["math score"]
print(Y)

print("Categories in 'gender' variable: ", end=" ")
print(df['gender'].unique())

print("Categories in 'race/ethnicity' variable: ", end=" ")
print(df['race/ethnicity'].unique())

print("Categories in 'parental level of education' variable:", end=" ")
print(df['parental level of education'].unique())

print("Categories in 'lunch' variable:  ", end=" ")
print(df['lunch'].unique())

print("Categories in 'test preparation course' variable:  ", end=" ")
print(df['test preparation course'].unique())

num_cols=X.select_dtypes(exclude='object').columns
cat_cols=X.select_dtypes(include='object').columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

num_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_cols),
        ("StandardScaler", num_transformer, num_cols),
    ]
)

x=preprocessor.fit_transform(X)
x.shape

#seperating the data into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,Y,test_size=0.2,random_state=23)

#creating an evaluate function to give all the metrics after training the model:
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)
    return mae, mse, rmse, r2

models = {
    "LR": LinearRegression(),
    "RR": Ridge(),
    "Lasso": Lasso(),
    "KNN": KNeighborsRegressor(),
    "DT": DecisionTreeRegressor(),
    "RF": RandomForestRegressor()

}
model_list = []
r2_list = []
for i in range(len(models)):
    model = list(models.values())[i]
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)
    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])
    print('Model Performance on Training Set:')
    print("-Root Mean squared Error:{:.4f}".format(model_train_rmse))
    print('-Mean Absolute Error:{:.4f}'.format(model_train_mae))
    print('-r2 Score:{:.4f}'.format(model_train_r2))
    print('--------------------------------')

    print('Model Performance on Testing Set:')
    print("-Root Mean squared Error:{:.4f}".format(model_test_rmse))
    print('-Mean Absolute Error:{:.4f}'.format(model_test_mae))
    print('-r2 Score:{:.4f}'.format(model_test_r2))
    r2_list.append(model_test_r2)

    print('=' * 35)
    print('\n')

    pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'R2 Score']).sort_values(by='R2 Score', ascending=False)
    print(pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'R2 Score']).sort_values(by='R2 Score', ascending=False))

    lin_model = LinearRegression(fit_intercept=True)
    lin_model = lin_model.fit(x_train, y_train)
    y_pred = lin_model.predict(x_test)
    score = r2_score(y_test, y_pred)*100
    print("Accuracy of Linear Regression model is %0.2f" % score)

    #plot y_pred vs y_test

plt.scatter(y_test, y_pred);
plt.xlabel("Actual Values");
plt.ylabel("Predicted Values");
plt.show()

sns.regplot(x=y_test, y=y_pred, ci=None, color="red");
plt.xlabel("Actual Values");
plt.ylabel("Predicted Values");
plt.show()

pred_df = pd.DataFrame({"Actual Values": y_test, "Predicted Values": y_pred, "Difference": y_test - y_pred})
print(pred_df.head(10))