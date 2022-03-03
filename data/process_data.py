import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    loading data from datasets:
    Arguments:
        :param messages_filepath:
        :param categories_filepath:
    :return  merged uncleaned dataframe df
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left=messages, right=categories, how='inner', left_on='id', right_on='id')
    return df

def clean_data(df):
    '''
    cleaning data and preparing for ML use, storing back in dataframe df
    :param df:
    :return: cleaned dataset df
    '''
    categories = df.categories.str.split(pat=';', n=-1, expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1].values

        # convert column from string to numeric
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df.related.replace(2, 1, inplace=True)

    return df



def save_data(df, database_filename):
    '''
    saving data into SQL database with given database filename

    :param df:
    :param database_filename:
    :return:
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)


def main():
    '''
    1) receives from user two datasets filepaths, then SQLdatabase path , where to store the cleaned prepared data
    2) loading and merging datasets
    3) cleaning datasets
    4) saving cleaned data into SQL database

    :return: cleaned preapered for ML usage SQL database
    '''

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()