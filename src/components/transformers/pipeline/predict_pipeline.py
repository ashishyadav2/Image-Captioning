from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import tensorflow as tf
from src.components.transformers.transformers_config import TransformersConfig
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_object
@dataclass
class PredictPipelineConfig:
    transformation_config = TransformersConfig()
    
class PredictPipeline:
    def __init__(self,model):
        self.config = PredictPipelineConfig()
        self.IMAGES_DIR = self.config.transformation_config.IMAGES_DIR
        self.IMAGE_SIZE = self.config.transformation_config.IMAGE_SIZE
        self.SEQ_LENGTH = self.config.transformation_config.SEQ_LENGTH
        self.index_lookup = None
        self.vectorization = load_object(os.path.join(self.config.transformers_config.ARTIFACT_DIR,"vecotorization.pkl"))
        self.val_data = None
        self.model = model
        
    def decode_and_resize(self,img_path):
        img = tf.io.read_file(os.path.join(self.IMAGES_DIR,img_path+".jpg"))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def vocab_config(self):
        vocab = self.vectorization.get_vocabulary()
        # mapping index to vocab tokens
        self.index_lookup = dict(zip(range(len(vocab)),vocab))
        max_decoded_sentence_length = self.SEQ_LENGTH - 1
        return max_decoded_sentence_length
    
    def generate_caption(self,image_path):
        validation_images, max_decoded_sentence_length = self.vocab_config()
        # sample_img = np.random.choice(validation_images)

        # # Read the image from the disk
        sample_img = os.path.join(self.config.transformation_config.IMAGES_DIR,image_path)
        sample_img = self.decode_and_resize(sample_img)
        # img = sample_img.numpy().clip(0, 255).astype(np.uint8)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()

        # Pass the image to the CNN
        img = tf.expand_dims(sample_img, 0)
        img = caption_model.cnn_model(img)
        encoded_img = caption_model.encoder(img,training=False)
        decoded_caption = "startseq "
        for i in range(max_decoded_sentence_length):
            # convert caption to integers
            tokenized_caption = self.vectorization([decoded_caption])[:,:-1]
            # create mask for tokenized caption
            mask = tf.math.not_equal(tokenized_caption,0)
            # give mask, tokens and image embedding to the decoder
            predictions = caption_model.decoder(
                tokenized_caption,encoded_img,training=False,
                mask = mask
            )
            sample_token_index = np.argmax(predictions[0,i,:])
            sample_token = self.index_lookup[sample_token_index]
            if sample_token == "endseq":
                break
            decoded_caption += " " + sample_token

        decoded_caption = decoded_caption.replace("startseq", "")
        decoded_caption = decoded_caption.replace("endseq", "").strip()
        print("Predicted Caption: ", decoded_caption)
        return decoded_caption
        