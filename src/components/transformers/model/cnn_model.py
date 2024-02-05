from src.exception import CustomException
from src.logger import logging
import os
import tensorflow as tf
from src.components.transformers.transformers_config import TransformersConfig

class CNNModel:
    def __init__(self):
        self.transformers_config = TransformersConfig()
        self.IMAGE_SIZE = self.transformers_config.IMAGE_SIZE
    
    def get_cnn_model(self):
        try:
            # load pre trained cnn model
            # include_top = False, do not import fully connected layer
            # use weights imagenet on that the model was trained on
            base_model = tf.keras.applications.efficientnet.EfficientNetB0(
                    include_top=False,
                    weights="imagenet",
                    input_shape=(*self.IMAGE_SIZE,3)
                )
            # freeze all the layers , the weights should not update
            base_model.trainable = False
            model_output = base_model.output
            # 2D -> 1D vector, image to 1D vector
            model_output = tf.keras.layers.Reshape((-1, model_output.shape[-1]))(model_output)
            # create model
            model = tf.keras.models.Model(base_model.input,model_output)
            logging.info("CNN model created")
            return model
        
        except Exception as e:
            logging.info("exception occured in loading cnn model")
            raise CustomException(e)