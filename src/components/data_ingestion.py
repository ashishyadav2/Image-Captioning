from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    DS_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.getcwd())), "artifacts", "dataset"
    )
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
    train, test = data_ingestion_obj.initiate_data_ingestion()
    print(train, test)
