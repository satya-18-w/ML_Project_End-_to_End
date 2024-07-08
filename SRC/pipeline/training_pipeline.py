

import os
import sys
from SRC.logger import logging
from SRC.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from SRC.components.data_transformation import DataTransformation
from SRC.components.data_ingestion import DataIngestion
from SRC.components.model_trainer import ModelTrainer






if __name__ == "__main__":
        
        obj=DataIngestion()
        train_data_path,test_data_path=obj.initiate_data_ingestion()
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        model_trainer=ModelTrainer()
        model_trainer.initiate_model_training(train_arr,test_arr)