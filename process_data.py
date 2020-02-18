import sys
from sqlalchemy import create_engine 
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """Load data
    
    INPUT
    messages_filepath - str, link to file
    categories_filepath - str, link to file
    
    OUTPUT
    df- pandas dataframe
    """
    ####from the previous ETL pipeline prep
    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    df=messages.merge(categories, on='id')
    
    return df
    


def clean_data(df):
    """ Cleaning data in df and transform categories part
    
    INPUT
    df - pandas dataframe
    
    OUTPUT
    df - cleaned pandas dataframe
    """
    ####from the previous ETL pipeline prep
    categories = df['categories'].str.split(pat=';',expand=True)
    row = categories.loc[0]
    category_colnames = row.str[:-2]
    print(category_colnames)
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """Saving dataframe to database path"""
    name='sqlite:///'+ database_filename
    engine = create_engine(name)
    df.to_sql('Disasters', engine, index=False)
    
    

def main():
    """Loads, cleans and saves the data in a databse"""
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