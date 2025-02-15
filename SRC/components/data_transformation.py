import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import StandardScaler # For Feature scaleing
from sklearn.preprocessing import OrdinalEncoder  # For performing ordinal Encoding

# Pipelines

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from SRC.exception import CustomException
from SRC.logger import logging
import os
from SRC.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()    
        
    def get_data_transformation_object(self):
        
        
        try:
            logging.info("Data Transformation Initiated")
            cut_category=["Fair","Good","Very Good","Premium","Ideal"]
            color_category=["D","E","F","G","H","I","J"]
            clarity_category=["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
            num=['carat', 'depth', 'table', 'x', 'y', 'z']
            cat=['cut','color','clarity']
            
            logging.info("Pipeline initiated")
            ## Numerical pipeline 
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    
                    ("scaler",StandardScaler())
                ]
            )


            ## Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("odianl",OrdinalEncoder(categories=[cut_category,color_category,clarity_category])),
                    
                    ("sclaer",StandardScaler())
                    
                ]
            )


            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,num),
                ("cat_pipeline",cat_pipeline,cat)
                
                
                
            ]    
            )
            logging.info("Pipeline Completed")
            
            return preprocessor
        
        except Exception as e:
            logging.info("Error in Data Transformation")        
            raise CustomException(e.sys)
    
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading the traing and test data 
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head: \n {train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")
            logging.info("Obtaining Preprocessor Object")
            
            target_column_name="price"
            drop_columns=[target_column_name,"id"]
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Applying preprocessing object  on training data ")
            preprocessor_obj=self.get_data_transformation_object()
            
            
            # Transformating using preprocessor obj
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            # Concatenating  target and imput
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_train_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            logging.info("Preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)
    
    # Run data