from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import tensorflow as tf
from src.components.transformers.transformers_config import TransformersConfig
from src.components.transformers.model.positional_embedding import PostionalEmbedding

@dataclass
class DecoderConfig:
    transformers_config = TransformersConfig()

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self,embed_dim,ffnn_dim,num_heads,**kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        # feed forward neural network dense dimensions
        self.ffnn_dim = ffnn_dim
        # num of heads for attention layer
        self.num_heads = num_heads

        self.decoder_config = DecoderConfig()
        self.SEQ_LENGTH = self.decoder_config.transformers_config.SEQ_LENGTH
        self.VOCAB_SIZE = self.decoder_config.transformers_config.VOCAB_SIZE
        self.EMBED_DIM = self.decoder_config.transformers_config.EMBED_DIM
        # will be used for target sequences (outputs)
        self.decoder_attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embed_dim,
            dropout = 0.2
        )

        # will be used cross attention layer (encoder_outputs,output_sequence_embeddings)
        self.decoder_attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embed_dim,
            dropout = 0.2
        )

        # feed forward neural network
        self.ffnn_layer_1 = tf.keras.layers.Dense(ffnn_dim,activation="relu")
        # linear layer
        self.ffnn_layer_2 = tf.keras.layers.Dense(embed_dim)

        self.layer_normalization_1 = tf.keras.layers.LayerNormalization()
        self.layer_normalization_2 = tf.keras.layers.LayerNormalization()
        self.layer_normalization_3 = tf.keras.layers.LayerNormalization()

        # to get positional embedding of output sequences
        self.embedding = PostionalEmbedding(
            sequence_length= self.SEQ_LENGTH,
            vocab_size= self.VOCAB_SIZE,
            embed_dim= self.EMBED_DIM
        )

        # softmax layer
        self.decoder_output = tf.keras.layers.Dense(self.VOCAB_SIZE,activation="softmax")

        # dropout layers
        self.dropout_1 = tf.keras.layers.Dropout(0.4)
        self.dropout_2 = tf.keras.layers.Dropout(0.4)
        self.supports_masking = True

    def call(self,inputs,encoder_outputs,training,mask=None):
        # get output embeddings
        inputs = self.embedding(inputs)
        # masks future tokens so that decoder cannot cheat
        casual_mask = self.get_casual_attention_mask(inputs)

        if mask is not None:
            # suppose mask shape is (5,5)
            padding_mask = tf.cast(mask[:,:,tf.newaxis], dtype = tf.int32) # after this operation (5,5,1)
            combined_mask = tf.cast(mask[:,tf.newaxis,:], dtype= tf.int32) # after this operation (5,1,5)
            # Returns the min of x and y (i.e. x < y ? x : y) element-wise.
            combined_mask = tf.minimum(combined_mask,casual_mask)

        # attention layer for output sequences
        # use combined attention mask
        attention_output_1 = self.decoder_attention_1(
            query = inputs,
            value = inputs,
            key = inputs,
            attention_mask = combined_mask,
            training = training
        )

        output_1 = self.layer_normalization_1(inputs+attention_output_1)

        attention_output_2 = self.decoder_attention_2(
            query = output_1,
            value = encoder_outputs,
            key = encoder_outputs,
            attention_mask = padding_mask,
            training = training
        )
        output_2 = self.layer_normalization_2(output_1+attention_output_2)

        ffnn_out = self.ffnn_layer_1(output_1)
        ffnn_out = self.dropout_1(ffnn_out,training=training)
        ffnn_out = self.ffnn_layer_2(ffnn_out)

        ffnn_out = self.layer_normalization_3(ffnn_out+output_2,training=training)
        ffnn_out = self.dropout_2(ffnn_out,training=training)
        preds = self.decoder_output(ffnn_out)
        return preds


    def get_casual_attention_mask(self,inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        # very important!
        # matrix of mask (lower triangular matrix containing 1s)
        mask = tf.cast(i>=j, dtype = "int32")
        # reshape mask matrix in (1,sequence_length,sequence_length)
        mask = tf.reshape(mask,(1,input_shape[1],input_shape[1]))
        # (64,1,1)
        mult = tf.concat([
            tf.expand_dims(batch_size, -1),
            tf.constant([1,1], dtype = tf.int32)
        ],axis = 0)
        # returns mask for each sequence in batch (60,40,40)
        return tf.tile(mask,mult)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'ffnn_dim': self.ffnn_dim,
            'num_heads': self.num_heads,
            'seq_length': self.SEQ_LENGTH,
            'vocab_size': self.VOCAB_SIZE,
            'embedd_dim': self.EMBED_DIM,
            'decoder_config': self.decoder_config
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)