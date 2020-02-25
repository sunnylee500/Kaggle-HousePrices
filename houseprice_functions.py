import numpy as np
import pandas as pd

def ratings_to_ord(df,col,inplace = False):
    '''
    This Function takes a dataframe and a column of that dataframe and returns and converts it to ordinal 
    df:
    col:
    inplace: 
    '''
    df[col] = df[col].fillna('Na')
    qual_ = {"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Na": 0}
    if inplace == False:
        return df[col].apply(lambda x: list(qual_.values())[list(qual_.keys()).index(x)])
    elif inplace == True:
        df[col] = df[col].apply(lambda x: list(qual_.values())[list(qual_.keys()).index(x)])

def outliers(df,outlier_column ,num_sd = 4,method = 'outlier_df', operator = 'any', drop_zeros = True):
    '''
    This function takes a dataframe and returns a dictionary that identifies the outliers of each column
    inputs:
    method: (length | Outlier_df)
    operator: (any | min_2 | all)
    '''
    outlier_dict = {}
    d = []
    full_outliers = []
    for col in outlier_column:
        if (df[[col]].dtypes[0] == np.int64()) or (df[[col]].dtypes[0] == np.float64()):
            lst_ = []
            outlier_dict[col] = lst_
            mean = df[col].mean()
            sd   = df[col].std()
            mean_no_z = df[col].drop(0).mean()
            sd_no_z = df[col].drop(0).std()
            if drop_zeros == False:
                outlier_bound_high = mean + sd*num_sd
                outlier_bound_low  = mean - sd*num_sd
            elif drop_zeros == True:
                outlier_bound_high = mean_no_z + sd_no_z*num_sd
                outlier_bound_low  = mean_no_z - sd_no_z*num_sd
            outliers_idx = df.index[df[col].apply(lambda x: (x < outlier_bound_low) or (x > outlier_bound_high))].tolist()        
            for i in outliers_idx:
                full_outliers.append(i)
            if method == 'length':
                outlier_dict[col] = [len(outliers_idx)]
            elif method == 'outlier_df':
                for i in outliers_idx:
                    d.append(df.iloc[i])
    if method == 'length':
        return pd.DataFrame.from_dict(outlier_dict,orient='index',columns=['Outlier_Count'])
    elif (method == 'outlier_df'):
        df_ = pd.DataFrame(d)
        if operator == 'any':
            return df_.drop_duplicates()
        elif operator == 'min_2':
            return df_[df_.duplicated()]
        elif operator == 'all':
            x = (pd.Series(full_outliers).value_counts(sort = False) == len(outlier_column.columns))
            x = pd.DataFrame(x,columns = ['all_'])
            x = x[x.all_ == True]
            return df_.merge(x,right_index = True, left_index = True,how = 'inner').drop('all_', axis = 1).drop_duplicates()

def outlier_selecter(df,outlier_column , num_sd = 3, min_unique = 20, drop_zeros = True,method = "dict"):
    '''
    creates a list of outliers to feed into the imputer
    '''
    outlier_dict = {}
    full_outliers = set()
    for col in outlier_column:
        if ((df[[col]].dtypes[0] == np.int64()) or (df[[col]].dtypes[0] == np.float64()))\
         and (df[col].nunique() >= min_unique):
            lst_ = []
            outlier_dict[col] = lst_
            mean = df[col].mean()
            sd   = df[col].std()
            mean_no_z = df[col].drop(0).mean()
            sd_no_z = df[col].drop(0).std()
            if drop_zeros == False:
                outlier_bound_high = mean + sd*num_sd
                outlier_bound_low  = mean - sd*num_sd
            elif drop_zeros == True:
                outlier_bound_high = mean_no_z + sd_no_z*num_sd
                outlier_bound_low  = mean_no_z - sd_no_z*num_sd
            outliers_idx = df.index[df[col].apply(lambda x: (x < outlier_bound_low) or (x > outlier_bound_high))].tolist()
            outlier_dict[col] = outliers_idx
            for i in outliers_idx:
                full_outliers.add(i)
    outlier_dict_nonz = {}
    for key, value in outlier_dict.items():
        if len(value) != 0:
            outlier_dict_nonz[key] = value
    if method == "dict":
        return outlier_dict_nonz
    elif method == "drop":
        return list(full_outliers)


def outlier_imputation(df_train,df_test,index_values, col = "",method = "drop_row",decimals = 0):
    '''
    df: dataframe
    index values: an integer or list of ints that indicate the rows that need to be imputed
    column: if mutatng using the values from a column input column name
    method : the method of imputation "drop", "mean", "median", "mode"
    decimals: num of decimals to include in the rounding, default 0
    '''
    if type(index_values) == int:
        index_values = [index_values]
    for idx in index_values:
        drop_ = []
        if method == "drop_row":
            if idx not in drop_:
                drop_.append(idx)
                df_test.drop(idx,inplace = True)
        elif method == "mean":
            df_test[col].iloc[idx] = round(df_train[col].mean(),decimals)
        elif method == "median":
            df_test[col].iloc[idx] = round(df_train[col].median(),decimals)
        elif method == "mode":
            df_test[col].iloc[idx] = round(df_train[col].mean(),decimals)
        elif method == "random":
            df_test[col].iloc[idx] = round(df_train[col].sample(random_state = 1).to_list()[0],decimals)

