# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Loads the data
    1. Loads the data from two files. The first is the messages and the second is the categories.
    2. Merge the two datasets

    Parameters
    ----------
    messages_filepath : str, mandatory
        File path to the messages file. It must be a CSV file

    categories_filepath : str, mandatory
        File path to the categories file. It must be a CSV file  """

    # 1. load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # 2. merge datasets
    df = pd.merge(messages, categories, on='id')

    return df

def clean_data(df):
    """ Cleans the data and returns a clean dataframe

    Parameters
    ----------
    messages_filepath : str, mandatory
        File path to the messages file. It must be a CSV file

    categories_filepath : str, mandatory
        File path to the categories file. It must be a CSV file  """

    # 1. Split categories into separate category columns.
    # take the categories column and split the values that are seperated by ; into a series
    # create a dataframe of the 36 individual category columns
    categories_series = df['categories'].str.split(";")
    categories_list = categories_series.to_list()
    categories_dataframe = pd.DataFrame(categories_list)

    # 2. select the first row of the categories dataframe
    row = categories_dataframe.iloc[0]

    # 3. remove all -1
    row = row.replace({'-1': ''}, regex=True)

    # remove all -0
    row = row.replace({'-0': ''}, regex=True)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row

    # rename the columns of `categories`
    categories_dataframe.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories_dataframe:
        # remove all the column words
        categories_dataframe[column] = categories_dataframe[column].replace({column: ''}, regex=True)

        # remove all the '-'
        categories_dataframe[column] = categories_dataframe[column].replace({'-': ''}, regex=True)

        # convert column from string to numeric
        categories_dataframe[column] = categories_dataframe[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_dataframe], axis=1, sort=False)

    # drop duplicates
    df = df.drop_duplicates()

    return df



def save_data(df, database_filename):
    """ takes dataframe and save as a sqlite database file

    Parameters
    ----------
    df : dataframe, mandatory
        Dataframe that will be written to a database file
    """

    irul = 'sqlite:///' + database_filename
    engine = create_engine(irul)
    try:
        df.to_sql('MessageClass', engine, index=False, if_exists='replace')
        print('Table created')
    except:
        print('Error creating table')


def main():
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()