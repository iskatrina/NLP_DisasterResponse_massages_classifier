
import pandas as pd
from sqlalchemy import create_engine

def ETL_pipeline(dataset_messages = 'messages.csv', dataset_categories = 'categories.csv'):
    # load  datasets
    messages = pd.read_csv(dataset_messages)
    categories = pd.read_csv(dataset_categories)
    df = pd.merge(left=messages,right=categories,how='inner',left_on='id',right_on='id')

    # create a dataframe of the 36 individual category columns
    categories_pre = df.categories.str.split(pat=';', n=-1, expand=True)

    #  extract a list of new column names for categories

    pre = categories.iloc[1].categories.replace('-0','')
    pre = pre.replace('-1','')
    column_names = pre.split(';')
    categories_pre.columns = column_names

    # Convert category values to just numbers 0 or 1.

    for column in categories_pre:
        # set each value to be the last character of the string
        categories_pre[column] = categories_pre[column].str[-1].values

        # convert column from string to numeric
        categories_pre[column] = categories_pre[column].astype(str)

    # Replace categories column in df with new category columns

    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories_pre],axis=1)
    df = df.drop_duplicates()

    # create/write SQL database
    engine = create_engine('sqlite:///..data/DesasterResponse.db', echo=False)
    df.to_sql('DesasterResponse', engine, index=False)



if __name__ == "__main__":
    ETL_pipeline()
