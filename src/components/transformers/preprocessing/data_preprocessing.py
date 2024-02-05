from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import tensorflow as tf
from src.components.transformers.transformers_config import TransformersConfig
import re
from src.utils import load_object, save_object
import numpy as np
from src.components.transformers.pipeline.train_pipeline import TrainPipeline
from src.components.transformers.pipeline.predict_pipeline import PredictPipeline

@dataclass
class PreprocessingConfig:
  transformers_config = TransformersConfig()
    
class Preprocessing:
  def __init__(self):
    self.config = PreprocessingConfig()
    self.image_caption_map = {}
    self.all_captions = []
    self.vectorization = None
    logging.info("preprocessing class")
    
  def flow(self):
    # image_caption_map = self.preprocess_caption(self.config.transformers_config.CAPTIONS_FILE)
    # save_object(os.path.join(self.config.transformers_config.ARTIFACT_DIR,"image_caption_map.pkl"),image_caption_map)
    self.image_caption_map = load_object(os.path.join(self.config.transformers_config.ARTIFACT_DIR,"image_caption_map.pkl"))
    self.get_all_captions()
    self.do_vectorization()
    logging.info("preprocessing flow()")
    
  def preprocess_caption(self,caption_path):
    image_caption_map = {}
    with open(caption_path,"r") as f:
      next(f)
      caption_file = f.read()

    for image_name_caption in caption_file.split("\n"):
      if len(image_name_caption)<2:
        continue
      line = image_name_caption.split(",")
      image_name,caption = line[0],line[1:]
      caption = " ".join(caption)
      caption = caption.lower()
      caption = re.sub(r'[^a-z\s]+',"",caption)
      caption = re.sub(r'\s+'," ",caption)
      caption = caption.strip()
      caption = "startseq " + caption + " endseq"
      image_name = image_name.split('.')[0]
      caption = " ".join([word for word in caption.split() if len(word)>1])
      if image_name not in image_caption_map:
        image_caption_map[image_name] = []
      image_caption_map[image_name].append(caption)
    logging.info("preprocessing preprocess_caption()")
    return image_caption_map

  def get_all_captions(self):
    for image_name,captions in self.image_caption_map.items():
      self.all_captions.extend(captions)
    logging.info("preprocessing get_all_captions()")
      
  
  def train_test_split(self,test_size=0.2,shuffle=True):
    image_ids = list(self.image_caption_map.keys())
    if shuffle:
      np.random.shuffle(image_ids)
    total_images_len = len(image_ids)
    test_samples_len = int(total_images_len * test_size)
    train_samples_len = int(total_images_len - test_samples_len)
    train_data = {image_id: self.image_caption_map[image_id] for image_id in image_ids[:train_samples_len]}
    test_data = {image_id: self.image_caption_map[image_id] for image_id in image_ids[train_samples_len:]}
    logging.info("preprocessing train_test_split()")
    return train_data,test_data
    
  def do_vectorization(self):
    self.vectorization = tf.keras.layers.TextVectorization(
        max_tokens = self.config.transformers_config.VOCAB_SIZE,
        output_mode = "int",
        output_sequence_length = self.config.transformers_config.SEQ_LENGTH
    )
    self.vectorization.adapt(self.all_captions)
    logging.info("preprocessing do_vectorization()")

  def preprocess_image(self,image):
    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.resize(img,self.config.transformers_config.IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img,tf.float32)
    return img

  def preprocess_inputs(self,image_id,captions):
    return self.preprocess_image(image_id), self.vectorization(captions)

  def make_dataset(self,data):
    images = list(map(lambda x: os.path.join(self.config.transformers_config.IMAGES_DIR,f"{x}.jpg"), list(data.keys())))
    captions = list(data.values())
    dataset = tf.data.Dataset.from_tensor_slices((images,captions))
    dataset = dataset.shuffle(self.config.transformers_config.BATCH_SIZE*8)
    dataset = dataset.map(self.preprocess_inputs,num_parallel_calls=self.config.transformers_config.AUTOTUNE)
    dataset = dataset.batch(self.config.transformers_config.BATCH_SIZE).prefetch(self.config.transformers_config.AUTOTUNE)
    logging.info("preprocessing make_dataset()")
    return dataset

  def get_dataset(self):
    train_data, val_data = self.train_test_split()
    train_dataset = self.make_dataset(train_data)
    val_dataset = self.make_dataset(val_data)
    logging.info("preprocessing get_dataset()")
    return train_dataset,val_dataset
  
  def get_vecotorization(self):
    save_object(os.path.join(self.config.transformers_config.ARTIFACT_DIR,"vecotorization.pkl"),self.vectorization)
    logging.info("preprocessing get_vectorization()")
    return self.vectorization

if __name__ == "__main__":
  preprocessing_obj = Preprocessing()
  preprocessing_obj.flow()
  vectorization = preprocessing_obj.get_vecotorization()
  train_dataset, val_dataset = preprocessing_obj.get_dataset()
  train_pipeline_obj = TrainPipeline(train_dataset,val_dataset)
  train_pipeline_obj.train_model()
  