from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
from src.components.transformers.transformers_config import TransformersConfig
import tensorflow as tf

@dataclass
class EncoderConfig:
    transformers_config = TransformersConfig()
    
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self,embed_dim,dense_dim,num_heads,**kwargs):
        super().__init__()
        # embed_dim is 512, originally used in transformers paper
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        # attention heads for attention layer
        self.num_heads = num_heads
        # attention layer, key_dim determines query and key dimensions
        self.encoder_attention = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embed_dim,
            dropout = 0.0
        )
        # layer normalization
        self.layer_normalization_1 = tf.keras.layers.LayerNormalization()
        self.layer_normalization_2 = tf.keras.layers.LayerNormalization()
        # feed forward neural network of encoder block
        self.encoder_dense = tf.keras.layers.Dense(embed_dim,activation="relu")


    def call(self,inputs,training,mask=None):
        # inputs goes to layer normalization
        # then dense layer (let output called as A)
        # then attention layer (let output called as B)
        # then A and B goes to layer normalization
        # mask is None because it is encoder layer no need to mask the token ids
        # training = training [boolean] used for training and inference
        inputs = self.layer_normalization_1(inputs)
        inputs = self.encoder_dense(inputs)

        # it is self attention because query=key=value
        attention_output = self.encoder_attention(
            query = inputs,
            key = inputs,
            value = inputs,
            attention_mask = None,
            training = training
        )

        encoder_output = self.layer_normalization_2(inputs+attention_output)
        return encoder_output

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'dense_dim': self.dense_dim,
            'num_heads': self.num_heads,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
