import pandas as pd
import numpy as np

def initialize(df):
    """
     transform some predictors to categorical, and then get dummies predictors, and then scale the predictors, and then split the x and y  
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




