import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import logging

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

#Console handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

#filehandler
log_file_path=os.path.join(log_dir,'feature_engineering.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str):
    '''loads data from csv file and convestes it into pandas dataframe'''
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('Data loaded and NANs filed from: %s',file_path)
        return df

    except pd.errors.ParserError as e:
        logger.error('Parse error while loading data : %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s',e)
        raise

def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features)->tuple:
    '''Applying tfidf to both train and test datasets gives out a tuple of train_df and test_df'''
    try:
        vectorizer=TfidfVectorizer(max_features=max_features)

        X_train=train_data['text'].values
        y_train=train_data['target'].values
        X_test=test_data['text'].values
        y_test=test_data['target'].values

        X_train_tfidf=vectorizer.fit_transform(X_train)
        X_test_tfidf=vectorizer.transform(X_test)

        train_df=pd.DataFrame(X_train_tfidf.toarray())
        train_df['label']=y_train
        
        test_df=pd.DataFrame(X_test_tfidf.toarray())
        test_df['label']=y_test

        logger.debug('bag of words created and data transformed')
        return train_df,test_df
    
    except Exception as e:
        logger.error('Unexpeected error while applying tfidf: %s', e)
        raise


def save_data(df:pd.DataFrame,file_path:str):
    '''saves 1 pandas df to the given file path as csv file'''
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug('File saved to file path: %s', file_path)
    except Exception as e:
        logger.error('Unexpected error while trying to save file: %s', e)
        raise


def main():
    try:
        max_features=50
        train_data=load_data('./data/interm_data/train_processed.csv')
        test_data=load_data('./data/interm_data/test_processed.csv')

        train_df,test_df=apply_tfidf(train_data,test_data,max_features=max_features)

        save_data(train_df,os.path.join('./data','processed','train_tfidf.csv'))
        save_data(test_df,os.path.join('./data','processed','test_tfidf.csv'))

    except Exception as e:
        logger.error('Failed to complete the feature_engineering process: %s', e)
        print(f'Error: {e}')

if __name__=='__main__':
    main()




