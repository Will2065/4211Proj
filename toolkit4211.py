import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def initialize(df):
    """
    transform some predictors to categorical; then split the x and y; and then get dummies predictors;  and then scale the predictors
     before conducting this function, you should generate the analyzed group, and drop the labels manually
    """
    df.brandID=df.brandID.astype('category')
    df.weekday=df.weekday.astype('category')
    y=df.sales
    x=df.drop('sales',axis=1)
    x=pd.get_dummies(x,drop_first=True)
    from sklearn.preprocessing import StandardScaler
    ss=StandardScaler(with_mean=False)
    x_stan=ss.fit_transform(x)
    x_stan=pd.DataFrame(x_stan,columns=x.columns)
    return x_stan, y

def PreOrganize(x_stan):
    """
     preorginize the pridictors, to construct 'sales~ a1+a2+...'
     input: standardized predictors value dataframe
    """
    predictors=list(x_stan.columns)
    pre_used='+ '.join(predictors)
    pre_used
    prefix='sales~'
    mulpre=prefix+''+pre_used
    return mulpre

def LeastSq_lr(df):
    """
    restore the dataset by combining x_stan and y, then conduct least square linear regression, and print the summary
    """
    import statsmodels.formula.api as smf
    x_stan,y=initialize(df)
    reveal=x_stan.copy()
    reveal['sales']=y.values
    results = smf.ols(PreOrganize(x_stan), data=reveal).fit()
    print(results.summary())
    return

def CV4LeastSq_lr(df):
    """
    conducting 10 folds cross validation based on Least Square Linear Regression, and compute the average MSE 
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    kf=KFold(n_splits=10)
    MSE=[]
    x_stan,y=initialize(df)
    for train_index, test_index in kf.split(x_stan):
        x_train= x_stan.iloc[train_index]
        x_test=x_stan.iloc[test_index]
        y_train=y.iloc[train_index]
        y_test=y.iloc[test_index]
        lm=LinearRegression()
        lm.fit(x_train,y_train)
        MSE.append(mean_squared_error(lm.predict(x_test),y_test))
    return np.mean(MSE)

def ridge_MSE(x_train_stan, x_test_stan,y_train,y_test,alpha):
    """
    compute the MSE_test
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    ridge=Ridge(alpha=alpha)
    ridge.fit(x_train_stan,y_train)
    MSE_test=mean_squared_error(ridge.predict(x_test_stan),y_test)
    return MSE_test

def CV4Ridge_lr(df,alpha):
    """
    conducting 10 folds cross validation based on Ridge Linear Regression, and compute the average MSE
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    kf=KFold(n_splits=10)
    MSE=[]
    x_stan,y=initialize(df)
    for train_index, test_index in kf.split(x_stan):
        x_train_stan= x_stan.iloc[train_index]
        x_test_stan=x_stan.iloc[test_index]
        y_train=y.iloc[train_index]
        y_test=y.iloc[test_index]
    MSE.append(ridge_MSE(x_train_stan, x_test_stan,y_train,y_test,alpha))
    return np.mean(MSE)

def para_plot(points,df,MinPara,MaxPara):
    """
    plot the MSE line graph to help to determine the best choice of parameter
    """
    p=points
    MSEs=[]
    A=np.linspace(MinPara,MaxPara,p)
    for i in A:
        MSEs.append(CV4Ridge_lr(df,i))
    plt.plot(A,MSEs)
    plt.show()
    return
    
def train_test(k, i):
    """
    from hzt
    """
    train = pd.read_csv('Data-train345.csv', index_col = 0)
    train1 = train.loc[train['Labels_' + str(k)] == i]
    train2 = train1.drop(['Labels_3', 'Labels_4', 'Labels_5'], axis = 1)
    X_train, y_train = tk.initialize(train2)
    return X_train, y_train

def RandomForest_MSE(x_train_stan, x_test_stan, y_train, y_test):
    """
    compute the MSE of a random Forest model
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    rf=RandomForestRegressor(n_estimators=1000, max_features='sqrt',random_state=25)
    rf.fit(x_train_stan, y_train)
    MSE_test=mean_squared_error(rf.predict(x_test_stan),y_test)
    return MSE_test

def CV4RandomForest(df):
    """
    compute the cross validation mse for random forest model
    input dataframe should be reprocessed group data
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    kf=KFold(n_splits=10)
    MSE=[]
    x_stan,y=initialize(df)
    for train_index, test_index in kf.split(x_stan):
        x_train= x_stan.iloc[train_index]
        x_test=x_stan.iloc[test_index]
        y_train=y.iloc[train_index]
        y_test=y.iloc[test_index]
        MSE_test=RandomForest_MSE(x_train, x_test, y_train, y_test)
        MSE.append(MSE_test)
    return np.mean(MSE)


