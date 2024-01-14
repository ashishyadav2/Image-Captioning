from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model import Model


@dataclass
class DataIngestionConfig:
    DS_PATH: str = os.path.join(os.getcwd(), "artifacts", "dataset")
    IMAGE_DIR_PATH: str = os.path.join(DS_PATH, "Images")
    CAPTIONS_FILE_PATH: str = os.path.join(DS_PATH, "captions.txt")


class DataIngestion:
    """
    this class reads image folder path and caption text file path,
    return (integer) number of image files in train set and rest in test set
    """

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, train_split_ratio=0.9):
        logging.info("Entered data ingestion method")
        try:
            logging.info("data ingestion initiated")
            total_images = len(os.listdir(self.data_ingestion_config.IMAGE_DIR_PATH))
            train_set = int(total_images * train_split_ratio)
            test_set = int(total_images * (1 - train_split_ratio))
            logging.info("data ingestion completed")
            return (train_set, test_set)
        
        except Exception as e:
            logging.info("Error occurred in data ingestion")
            CustomException(e)


if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    train_split, test_split = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation(
        train_split,
        test_split,
        data_ingestion_obj.data_ingestion_config.IMAGE_DIR_PATH,
        data_ingestion_obj.data_ingestion_config.CAPTIONS_FILE_PATH,
    )
    
    data_transformation_obj.initiate_data_transformation()
    data_transformation_config = data_transformation_obj.get_config()
    train_set,test_set = data_transformation_obj.train_test_split()
    
    model_obj = Model(data_transformation_config.max_caption_len,data_transformation_config.vocab_size)
    model = model_obj.get_model()
    
    model_trainer_obj = ModelTrainer(data_transformation_config,train_set,test_set,model)
    # model_trainer_obj.plot_model_structure()
    model_trainer_obj.train_model(epochs=25)
