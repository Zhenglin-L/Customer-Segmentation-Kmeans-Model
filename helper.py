# import libraries here; add more as necessary
import numpy as np
import pandas as pd


# replace 0/9 with -1 for some columns, to change all unknown cell value into -1
def replace_val(df,target_columns,target_val,new_val):
    
    converted_df = df[target_columns].replace(target_val,new_val)
    df = df.drop(target_columns, axis=1)
    prepared_df = pd.concat([df,converted_df],axis=1)
    
    return prepared_df



def find_mean(df):
    ''' replace -1 with mean of the column
        find mean for each column
        return a list of the mean of each column
    '''
    mean = []
    for column in df.columns:
        count_non_neg_one = 0
        sum_val = 0
        column_mean = 0
        for cell in df[column]:
            if cell!=-1:
                sum_val = sum_val + cell
                count_non_neg_one = count_non_neg_one + 1
        if count_non_neg_one == 0:
            column_mean = 0
        else:
            column_mean = sum_val/count_non_neg_one
        mean.append(column_mean)
    return mean                        



def replace_unknown_to_mean(df):
    mean = find_mean(df)
    i=0
    for column in df.columns:
        df[column] = df[column].replace(-1,mean[i])
        i=i+1
    return df








