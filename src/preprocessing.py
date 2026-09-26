import logging 
import os
import nltk
import wordcloud
import pandas as pd
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('data_preprocssing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_dir=os.path.join(log_dir,'data_preprocessing.log')
filehandler=logging.FileHandler(log_file_dir)
filehandler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(filehandler)

ps=PorterStemmer()
def transform_text(text:str):
    '''Removes special characters form text and stems the tokens to return transformed text'''

    text=text.lower()

    text=nltk.word_tokenize(text)

    #removes special characters
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()

    #removing stopwords(i.e junction words) and punctuations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()

    #stemming (makes similar words as same tokens)
    for i in text:
        y.append(ps.stem(i))

    #returns everything in form of string
    return ' '.join(y)


def preprocess_data(df:pd.DataFrame,text_collum='text',target_collum='target')->pd.DataFrame:
    '''Encodes target,removes duplicates,transforms text returns back dataframe'''
    try:
        logger.debug('Preprocessing started: %s')
        encoder=LabelEncoder()

        #encoding target collum
        df[target_collum]=encoder.fit_transform(df[target_collum])
        logger.debug('Target collum encoded')

        #droping duplicate values in df
        df=df.drop_duplicates()
        logger.debug('Duplicates droped')

        #transforming text into a suitable form
        df.loc[:,text_collum]=df[text_collum].apply(transform_text)
        logger.debug('Text transformed')

        #returning the transformed dataframe
        return df
        

    except KeyError as e:
        logger.error('Collum missing error: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error in normalizing: %s',e)
        raise

def main(text_collum='text',target_collum='target'):
    'Loads data from directory processes it and saves the processed data to ./data/interm_data'

    try:
        train_df=pd.read_csv('./data/raw/train.csv')
        test_df=pd.read_csv('./data/raw/test.csv')
        logger.debug('Data for processing loaded sucessfully')

        final_train_df=preprocess_data(df=train_df,text_collum='text',target_collum='target')
        final_test_df=preprocess_data(df=test_df,text_collum='text',target_collum='target')

        data_path=os.path.join('./data','interm_data')
        os.makedirs(data_path,exist_ok=True)

        final_train_df.to_csv(os.path.join(data_path,'train_processed.csv'))
        final_test_df.to_csv(os.path.join(data_path,'test_processed.csv'))
        logger.debug('Processedd Data to %s, data_path')

    except FileNotFoundError as e:
        logger.error('File not found error: %s', e)
        raise

    except pd.errors.EmptyDataError as e:
        logger.error('Empty data error: %s', e)
        raise
    
    except Exception as e:
        logger.error('Unexpected error in data transformation process: %s',e)
        raise

        

if __name__=='__main__':
    main()




