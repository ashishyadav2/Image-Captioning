from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import tensorflow as tf

@dataclass
class TransformersConfig:
    IMAGE_SIZE = (299, 299)
    # number of words in the text
    VOCAB_SIZE = 10000
    # maximum output of the sequence
    SEQ_LENGTH = 40
    # fixed size representation
    EMBED_DIM = 512
    # feed forward neural network dimension
    FF_DIM = 512
    BATCH_SIZE = 64
    EPOCHS = 100
    # used for optimizing input pipelines, prefetch etc
    AUTOTUNE = tf.data.AUTOTUNE
    ARTIFACT_DIR = os.path.join(os.getcwd(),"artifacts")
    IMAGES_DIR = os.path.join(ARTIFACT_DIR,"dataset","Images")
    CAPTIONS_FILE = os.path.join(ARTIFACT_DIR,"dataset","captions.txt")