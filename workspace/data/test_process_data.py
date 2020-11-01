# import libraries
from process_data import load_data
from process_data import clean_data
from process_data import save_data
import pandas as pd

def test_load_data():
    assert(load_data('disaster_messages.csv', 'disaster_categories.csv').shape == (26386, 5))


def test_clean_data():
    df1 = load_data('disaster_messages.csv', 'disaster_categories.csv')
    df2 = clean_data(df1)

    assert (df2.shape == (26216, 40))

def test_write_to_db():
    df1 = load_data('disaster_messages.csv', 'disaster_categories.csv')
    df2 = clean_data(df1)
    database_filepath = 'sqlite:///DisasterResponsetest.db'
    save_data(df2, database_filepath)
    # load data from database
    df3 = pd.read_sql_table('MessageClass', database_filepath)

    assert (df3.shape == (26216, 40))
