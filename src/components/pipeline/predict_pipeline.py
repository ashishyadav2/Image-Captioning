from src.logger import logging
from src.exception import CustomException
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from src.components.data_transformation import FeatureExtractionModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from src.utils import load_object, load_keras_model


class PredictPipeline:
    def __init__(self, model_name):
        data_transformation_config = load_object(
            os.path.join(".", "artifacts", "data_transformation_config.pkl")
        )
        self.tokenizer = data_transformation_config.tokenizer
        self.max_caption_len = data_transformation_config.max_caption_len
        self.image_caption_map = load_object(
            data_transformation_config.image_caption_map
        )
        self.image_feature_map = load_object(
            data_transformation_config.image_feature_map
        )
        self.model = load_keras_model(
            os.path.join(".", "artifacts", model_name)
        )
        self.index_word_map = {
            index: word for word, index in self.tokenizer.word_index.items()
        }
        self.feature_extractor_model = FeatureExtractionModel()

    def index_to_word(self, idx):
        return self.index_word_map[idx]

    def predict_caption(self, image_feature):
        try:
            output = "startseq"
            for i in range(self.max_caption_len):
                seq = self.tokenizer.texts_to_sequences([output])[0]
                seq = pad_sequences([seq], self.max_caption_len)
                y_pred = self.model.predict([image_feature, seq], verbose=0)
                y_pred = np.argmax(y_pred)
                word = self.index_to_word(y_pred)
                if word is None:
                    break
                output = output + " " + word
                if word == "endseq":
                    break
            logging.info("caption predicted")
            return output

        except Exception as e:
            logging.info("error in predicting caption")
            raise CustomException(e)

    def predict(self, image_name):
        img_path = os.path.join(".", "static", "uploads", image_name)
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        feature_model = self.feature_extractor_model.get_model()
        image_feature = feature_model.predict(img, verbose=0)
        predicted_caption = self.predict_caption(image_feature)
        predicted_caption = " ".join(predicted_caption.split()[1:-1]).capitalize()
        return predicted_caption
