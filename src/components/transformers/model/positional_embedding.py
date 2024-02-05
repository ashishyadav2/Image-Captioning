from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import tensorflow as tf


class PostionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,sequence_length,vocab_size,embed_dim,**kwargs):
        super().__init__(**kwargs)
        # convert words to numbers
        self.token_embedding = tf.keras.layers.Embedding(input_dim= vocab_size, output_dim=embed_dim)
        # generate postional embedding to learn about the position of words in a sentence, since we are not using RNN this will helpp the model to learn about the postion of the words
        self.position_embedding = tf.keras.layers.Embedding(input_dim = sequence_length, output_dim = embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # converting embed_dim to float datatype and taking sqrt (later to used to scale the values so that they should not be so large)
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self,inputs):
        # inputs come in batches (batch_size, rows, cols) so get last dimension i.e length of the sequence
        length = tf.shape(inputs)[-1]
        # same as range in python start,end , delta is increment
        # same as range(0,length,1)
        positions = tf.range(start=0,limit=length,delta=1)
        # get token embeddings
        embedded_tokens = self.token_embedding(inputs)
        # scale the embedded tokes
        embedded_tokens = embedded_tokens * self.embed_scale
        # get positional embeddings
        embedded_positions = self.position_embedding(positions)
        # combine token embeddings + positional embeddings
        return embedded_tokens + embedded_positions

    def compute_mask(self,inputs,mask=None):
        # returns boolean tensor where ever inputs value is not equal to 0 returns true else False
        return tf.math.not_equal(inputs,0)

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)