from src.logger import logging
from src.exception import CustomException
import os
from dataclasses import dataclass, field
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image, text
from src.utils import save_object
import re
from src.utils import load_object, save_object


@dataclass
class DataTransformationConfig:
    DS_PATH: str = os.path.join(os.getcwd(), "artifacts")
    image_feature_map: str = os.path.join(DS_PATH, "image_feature_map.pkl")
    image_caption_map: str = os.path.join(DS_PATH, "image_caption_map.pkl")
    all_captions: str = os.path.join(DS_PATH, "all_captions.pkl")
    max_caption_len: int = 0
    vocab_size: int = 0
    tokenizer = None
    image_ids = []


class FeatureExtractionModel:
    def __init__(self):
        vgg16_model = VGG16(weights="imagenet")
        feature_extractor_model = models.Model(
            inputs=vgg16_model.inputs, outputs=vgg16_model.layers[-2].output
        )
        return feature_extractor_model


class DataTransformation:
    def __init__(self, train_split, test_split, IMAGE_DIR, CAPTIONS_FILE_PATH):
        self.data_transformation_config = DataTransformationConfig()
        self.train_split = train_split
        self.test_split = test_split
        self.IMAGE_DIR = IMAGE_DIR
        self.CAPTIONS_FILE_PATH = CAPTIONS_FILE_PATH

    def initiate_data_transformation(self):
        logging.info("data transformation initiated")
        # self.extract_feature(self.IMAGE_DIR)
        self.preprocess_caption(self.CAPTIONS_FILE_PATH)
        self.preprocessing_text()
        self.tokenizing_and_fitting_text()

    def extract_feature(self, IMAGE_DIR):
        """
        this function extracts feature from the image uisng vgg16 model and creates
        image id feature map
        """
        logging.info("feature extraction started")
        image_feature_map = {}
        feature_extractor_model = FeatureExtractionModel()
        try:
            for image_name in os.listdir(IMAGE_DIR):
                img_path = os.path.join(IMAGE_DIR, image_name)
                img = image.load_img(img_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                img = preprocess_input(img)
                img_id = image_name.split(".")[0]
                extracted_feature = feature_extractor_model.predict(img, verbose=0)
                image_feature_map[img_id] = extracted_feature
            save_object(
                self.data_transformation_config.image_feature_map, image_feature_map
            )
            logging.info("Image feature map object saved")

        except Exception as e:
            logging.info("error in image feature extraction")
            raise CustomException(e)

    def preprocess_caption(self, caption_path):
        """
        this function cleans captions by removing extra spaces, removing everything except the lowercase letters, and anything whose length is greater than 1.
        Also make image id caption map. each image id contains 5 captions which is stored as list of 5 captions
        """
        logging.info("caption preprocessing started")
        image_caption_map = {}
        with open(caption_path, "r") as f:
            next(f)
            caption_file = f.read()
        try:
            for image_name_caption in caption_file.split("\n"):
                if len(image_name_caption) < 2:
                    continue
                line = image_name_caption.split(",")
                image_name, caption = line[0], line[1:]
                caption = " ".join(caption)
                caption = caption.lower()
                caption = re.sub(r"[^a-z\s]+", "", caption)
                caption = re.sub(r"\s+", " ", caption)
                caption = caption.strip()
                caption = "startseq " + caption + " endseq"
                image_name = image_name.split(".")[0]
                caption = " ".join([word for word in caption.split() if len(word) > 1])
                if image_name not in image_caption_map:
                    image_caption_map[image_name] = []
                image_caption_map[image_name].append(caption)

            save_object(
                self.data_transformation_config.image_caption_map, image_caption_map
            )
            logging.info("Image caption map object saved")

        except Exception as e:
            logging.info("error in caption cleaning")
            raise CustomException(e)

    def preprocessing_text(self):
        """
        this function makes a list of all captions adn save it in a pkl file
        """
        logging.info("making list of all captions")
        image_caption_map = load_object(
            self.data_transformation_config.image_caption_map
        )
        all_captions = []
        try:
            for image_name, captions in image_caption_map.items():
                all_captions.extend(captions)

            save_object(self.data_transformation_config.all_captions, all_captions)
            logging.info("all captions completed")

        except Exception as e:
            logging.info("error occurred in making all captions ")
            raise CustomException(e)

    def tokenizing_and_fitting_text(self):
        logging.info("tokenizing and fitting on text started")
        all_captions = load_object(self.data_transformation_config.all_captions)
        self.data_transformation_config.tokenizer = text.Tokenizer()
        self.data_transformation_config.tokenizer.fit_on_texts(all_captions)
        self.data_transformation_config.vocab_size = len(self.data_transformation_config.tokenizer.word_index) + 1
        self.data_transformation_config.max_caption_len = max(
            len(caption.split()) for caption in all_captions
        )
        logging.info("tokenizing and fitting on text completed")

    def train_test_split(self):
        logging.info("spliting into train and test set")
        image_caption_map = load_object(
            self.data_transformation_config.image_caption_map
        )
        self.data_transformation_config.image_ids = list(image_caption_map.keys())
        train_set = self.data_transformation_config.image_ids[: self.train_split]
        test_set = self.data_transformation_config.image_ids[self.train_split :]
        return (train_set, test_set)

    def get_config(self):
        return self.data_transformation_config
        
