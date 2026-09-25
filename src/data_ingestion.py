import logging
import yaml
import os
import pandas as pd
from sklearn.model_selection import train_test_split

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#Setting up logger
logger=logging.getLogger('data ingestion')
logger.setLevel('DEBUG')

#console handler
console_handeler=logging.StreamHandler()
console_handeler.setLevel('DEBUG')

#log_handler
log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#formater
formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handeler.setFormatter(formater)
file_handler.setFormatter(formater)

#adding handlers to the logger which we created
logger.addHandler(console_handeler)
logger.addHandler(file_handler)

def load_parms(param_path:str)->dict:
    try:
        with open(param_path,'r') as file:
            parms=yaml.safe_load(file)
        logger.debug('Parameters retived from %s',param_path)
        return parms
    except FileNotFoundError:
        logger.error('File not found: %s',param_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Yaml error : %s', e)
        raise
    except Exception as e :
        logger.error('Unexpected error in loading params: %s', e)
        raise


def load_data (data_url:str)-> pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug('Data loaded successfully from %s', data_url)
        return df
    
    except pd.errors.ParserError as e:
        logger.error('Failed to parse csv file : %s', e)
        raise

    except Exception as e:
        logger.error('Unexpected error in loading data: %s', e)
        raise


def data_prepocessing(df:pd.DataFrame)-> pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug('Data cleaned : %s')
        return df

    except KeyError as e:
        logger.error('Missing colloum in df: %s', e)
        raise

    except Exception as e:
        logger.error('Unexpected error in data_prepocessing: %s', e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str):
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('Data saved to: %s', raw_data_path )

    except Exception as e:
        logger.error('Unexpected error in save_data: %s', e)
        raise

def main():
    try:
        parms=load_parms(param_path='params.yaml')
        test_size=parms['data_ingestion']['test_size']
        data_url='https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df=load_data(data_url)
        final_df=data_prepocessing(df)

        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=2)
        save_data(train_data,test_data,data_path='./data')

    except Exception as e:
        logger.error('Unexpected error in main: %s', e)


if __name__=='__main__':
    main()

