import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from SRC.exception import CustomException
from SRC.logger import logging





def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)        
    
    
    
    
def evaluate_model(X_train,Y_train,X_test,Y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.value())[i]
            # Train Model
            model.fit(X_train,Y_train)
            
            Y_test_pred=model.predict(X_test)
            
            
            test_model_score=r2_score(Y_test,Y_test_pred)
            report[list(models.key())[i]] = test_model_score
            
            
        return report
    except Exception as e:
        logging.info("Exception occured during model training")
        raise CustomException(e,sys)    
    
    
    
    def load_object(file_object):
        try:
            with open(file_path,"rb") as file_obj:
                return pickle.load(file_obj)
            
        except Exception as e:
            logging.info("Exception Occured in load_object function utills")
            raise CustomException(e,sys)
        