import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from SRC.exception import CustomException
from SRC.logger import logging
from SRC.utils import save_object,evaluate_model
from dataclasses import dataclass
import sys
import os


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting Dependent and independent variables from train and test array")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:-1]
            )    
        
            models={
                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "ElasticNet":ElasticNet()
            }    
            
            
            model_report:dict=evaluate_model(X_train,Y_train,X_test,Y_test,models)
            
            print(model_report)
            print("\n===========================================================================================")
            logging.info(f"Model Report :{model_report}")
            # To get best model score from dictionary
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            print(f"Best Model found , Model Name : {best_model_name} , R2_score : {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            
        except Exception as e:
            logging.info("Exception occured in the Model training")
            raise CustomException(e,sys)
        
        
        
        
        
            