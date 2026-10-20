import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,roc_auc_score,recall_score
import pickle
import json

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_evaluation.log')
filehandler=logging.FileHandler(log_file_path)
filehandler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(filehandler)


def load_model(file_path:str):
    try:
        with open(file_path,'rb')as file:
            model=pickle.load(file)
        
        logger.debug('Model loaded successfully from : %s',file_path)
        return model
    
    except FileNotFoundError as e:
        logger.error('Filenotfound error in load_model function: %s', e)
        raise

    except Exception as e:
        logger.error('Unexpected eror while trying to load model: %s', e)
        raise


def load_test_df(test_df_path:str)->pd.DataFrame:
    try:
        
        df=pd.read_csv(test_df_path)

        logger.debug('test_dataframe loaded suscessfully from: %s',test_df_path)    
        return df

    except pd.errors.ParserError as e:
        logger.error('Parse error while trying to load test dataframe: %s', e)
        raise

    except Exception as e:
        logger.error('Unexpected error while trying to load test df : %s', e)
        raise

def evaulation(model,X_test:np.ndarray,y_test:np.ndarray)->dict:
    try:
        y_pred=model.predict(X_test)
        y_pred_proba=model.predict_proba(X_test)[:,1]

        accuracy=accuracy_score(y_true=y_test,y_pred=y_pred)
        precision=precision_score(y_test,y_pred)
        roc_auc=roc_auc_score(y_test,y_pred_proba)
        recall=recall_score(y_test,y_pred)

        scores={'accuracy_score':accuracy,
                'precision_socre':precision,
                'recall_score':recall,
                'roc_auc_score':roc_auc}

        logger.debug('Model evaluation done and scores collected succesfully')
        
        return scores

    except Exception as e:
        logger.error('Unexpectied error during model evaulation: %s', e)
        raise

def save_metrics(metrics: dict,file_path:str):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)

        logger.debug('Metrics saved sucessfully inside: %s', file_path)

    except Exception as e:
        logger.error('Unexpected error while trying to save_metrics: %s', e)
        raise


def main():
    try:
        model=load_model('./models/model.pkl')
        df=load_test_df('./data/processed/test_tfidf.csv')

        X_test=df.iloc[:,:-1].values
        y_test=df.iloc[:,-1].values

        metrics_dict=evaulation(model,X_test,y_test)

        save_metrics(metrics_dict,'reports/metrics.json')

        logger.debug('Whole process completed sucessfully')

    except Exception as e:
        logger.error('Unexpected error during main process : %s', e)
        print(f'Error: {e}')

if __name__=='__main__':
    main()






    