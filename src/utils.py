import pickle
from src.exception import CustomException
from src.logger import logging
import os
from tensorflow.keras import models


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info(f"Error in saving object")
        raise CustomException(e)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info(f"Error in loading object")
        raise CustomException(e)

def save_model(self,file_path):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            self.model.save(file_path)
            logging.info("model saved")
            
        except Exception as e:
            logging.info(f"Error in saving object")
            raise CustomException(e)
        
def load_keras_model(self,file_path):
    try:
        model = models.load_model(file_path)
        logging.info("model loaded")
        return model
        
    except Exception as e:
        logging.info(f"Error in loading object")
        raise CustomException(e)