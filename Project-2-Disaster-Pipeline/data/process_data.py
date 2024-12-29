import sys
import pandas as pd
from sqlalchemy import create_engine

# ------------------------------------------------------------------------------------

def load_data(messages_filepath, categories_filepath):
    """ Load CSV data for both messages and categories and merges them

    Args:
        messages_filepath (str): Path of messages CSV
        categories_filepath (_type_): Path of categories CSV
    
    Returns:
        df (DataFrame): Merged datafrrame containing messages and categories
    """

    # Read in CSV files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets using 'id' column
    df = messages.merge(categories, on='id')

    return df

# ------------------------------------------------------------------------------------

def clean_data(df):
    """ Cleans data by splitting out categories column into separate columns, convering to binary values, and dropping duplicate rows
    
    Args:
        df (dataframe): Merged dataframe output from load_data

    Returns:
        df: Cleaned dataframe with
    """

    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    category_cols = [val.split('-')[0] for val in categories.iloc[0,:]]
    categories.columns = category_cols
    for col in categories:
        categories[col] = categories[col].str[-1]
        categories[col] = categories[col].astype(int)
    
    # Drop original categories column and replace with new set of columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    return df

# ------------------------------------------------------------------------------------

def save_data(df, database_filename):
    """ Save dataframe to SQLite database

    Args:
        df (dataframe): Cleaned dataframe output from clean_data()
        database_filename (str): Path to save SQLite database
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')

# ------------------------------------------------------------------------------------

def main():
    """ Main function that runs the full ETL pipeline

    Takes 3 argumnts:
        1. messages_filepath: Path to messages CSV
        2. categories_filepath: Path to categories CSV
        3. database_filepath: Path to save output SQLite database
        
    Example: python process_data.py disaster_messages.csv disaster_categories.csv disaster_data.db
    """
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

# ------------------------------------------------------------------------------------