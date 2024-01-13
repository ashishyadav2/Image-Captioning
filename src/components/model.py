from src.logger import logging
from src.exception import CustomException
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image,text
from dataclasses import dataclass

@dataclass
class ModelConfig:
    max_caption_len = 34
    vocab_size = 8000

class Model:
    def __init__(self):
        self.model_config = ModelConfig()
        max_caption_len = self.model_config.max_caption_len
        vocab_size = self.model_config.vocab_size
        
        input1 = layers.Input(shape=(4096,))
        dropout_layer_fe = layers.Dropout(0.5)(input1)
        fc3 = layers.Dense(units=256,activation='relu')(dropout_layer_fe)
        
        input2 = layers.Input(shape=(max_caption_len,))
        embedding_layer = layers.Embedding(input_dim = vocab_size, output_dim = 256, mask_zero=True)(input2)
        dropout_layer_en = layers.Dropout(0.5)(embedding_layer)
        lstm_layer = layers.LSTM(256)(dropout_layer_en)
        
        fc3 = layers.add([fc3,lstm_layer])
        fc4 = layers.Dense(units = 256,activation='relu')(fc3)
        fc5 = layers.Dense(units=vocab_size,activation = "softmax")(fc4)

        model = models.Model(inputs = [input1,input2], outputs=fc5)
        model.compile(loss='categorical_crossentropy',optimizer='adam')

        # plot_model(model,show_shapes=True)
        return model