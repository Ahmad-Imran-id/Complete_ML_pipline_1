import os
import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')

#console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# filehandler
log_file_path=os.path.join(log_dir,'model_training.log')
file_handler=logging.FileHandler(log_file_path)

#formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#adding handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path:str)->tuple:
    try:
        df=pd.read_csv(file_path)
        X_train=df.iloc[:,:-1].values
        y_train=df.iloc[:,-1].values
        
        logger.debug('Data loaded from %s with X_train shape: %s , y_train shape: %s ',file_path,X_train.shape,y_train.shape)
        return X_train,y_train
    except pd.errors.ParserError as e:
        logger.error('fialed to parse the csv file : %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('FIle not found error duing loading_data: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading the model: %s', e)
        raise


def model_training(X_train:np.ndarray,y_train:np.ndarray,params:dict):
    try:
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        clf.fit(X_train,y_train)
        logger.debug('Model trained sucessfully')
        return clf

    except ValueError as e:
        logger.error('Value error during model training: %s', e)
        raise
    
    except Exception as e:
        logger.error('Unexpected error while training the model: %s', e)
        raise

def save_model(model,filepath:str)->None:
    try:
        os.makedirs(os.path.dirname(filepath),exist_ok=True)
        with open(filepath,'wb')as file:
            pickle.dump(model,file)

        logger.debug('FIle saved to %s', filepath)

    except FileNotFoundError as e:
        logger.error('FIle not found error during saving model: %s', e)
        raise 

    except Exception as e:
        logger.error('Unexpected error while loading the model: %s', e)
        raise


def main():
    try:
        params=load_params(params_path='params.yaml')['model_training']


        X_train,y_train=load_data('./data/processed/train_tfidf.csv')

        clf=model_training(X_train,y_train,params)

        save_path='models/model.pkl'
        save_model(clf,save_path)
        
        logger.debug('model succesfully trained and saved')
    
    except Exception as e:
        logger.error('Error during main function: %s', e)
        raise

if __name__=='__main__':
    main()

