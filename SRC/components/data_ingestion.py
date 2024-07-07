import os
import sys
from SRC.logger import logging
from SRC.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# Initialize the Data Ingestion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    
    
    

## Create a class for Data Ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method starts")    
        
        try:
            
            df=pd.read_csv("./notebook/gemstone.csv")
            logging.info("Dataset read as pandas dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Train test split")
            train_set,test_set=train_test_split(df,test_size=0.3)        
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data Ingestion is complete")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Exception occured at Data Ingestion Stage")
            raise CustomException(e,sys)
        
        
    
    ## Run Data Ingestion
    if __name__ == "__main__":
        
        try:
            
            obj = DataIngestion()
            train_data, test_data = obj.initiate_data_ingestion()
        except CustomException as e:
            logging.error(f"Failed to complete data ingestion: {e}")