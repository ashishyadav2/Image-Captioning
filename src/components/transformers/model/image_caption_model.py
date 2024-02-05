from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import tensorflow as tf
from src.components.transformers.transformers_config import TransformersConfig


class ImageCaptionModel(tf.keras.Model):
    def __init__(
        self,
        cnn_model,
        encoder,
        decoder,
        num_captions_per_image = 5,
        image_aug = None
        ):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self,y_true,y_pred,mask):
        # loss and mask are vectors/tensors
        loss = self.loss(y_true, y_pred) # calculate loss
        mask = tf.cast(mask,dtype = loss.dtype) # convert mask to data type of loss calculated
        loss *= mask # remove contribution of padding tokens so loss of padded tokens will become zero and non padded tokens retains
        return tf.reduce_sum(loss) / tf.reduce_sum(mask) # normalized loss by removing padding token contribution

    def calculate_accuracy(self,y_true,y_pred, mask):
        # get index of maximum probability along 3rd axis (batch,rows,cols) 0, 1, 2
        # returns boolean if index and y_true are equal
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis = 2))
        # Keeps only the accuracy values for masked elements, effectively ignoring those where the mask is False.
        accuracy = tf.math.logical_and(mask,accuracy)
        accuracy = tf.cast(accuracy, dtype = tf.float32)
        mask = tf.cast(mask, dtype = tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self,img_embed, batch_seq, training=True):
        # get image embeddings
        encoder_out = self.encoder(img_embed, training=training)
        #batch_seq_input = batch_seq[:, :-1]: Shifts captions to the left, creating input sequences for the decoder.
        batch_seq_input = batch_seq[:,:-1]
        # batch_seq_true = batch_seq[:, 1:]: Shifts captions to the right, forming target sequences for comparison.
        batch_seq_true = batch_seq[:, 1:]
        # Generates a boolean mask to ignore padding tokens in loss and accuracy calculations.
        mask = tf.math.not_equal(batch_seq_true, 0)

        batch_seq_pred = self.decoder(
            batch_seq_input, encoder_out, training=training, mask = mask
        )
        loss = self.calculate_loss(batch_seq_true,batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self,batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        if self.image_aug:
            batch_img = self.image_aug(batch_img)

        img_embed = self.cnn_model(batch_img)

        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq[:,i,:], training=True)

                batch_loss += loss
                batch_acc += acc

            # getting encoder and decoder weights and adding them
            train_vars = (self.encoder.trainable_variables+self.decoder.trainable_variables)

            gradients = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(gradients,train_vars))

        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result()
        }

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        img_embed = self.cnn_model(batch_img)

        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(
                img_embed, batch_seq[:, i, :], training=False
            )

            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)

        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def get_config(self):
        base_config = super().get_config()
        config = {
            "loss": self.loss,
            "optimizer": self.optimizer.get_config(),
            "metrics": [metric.name for metric in self.metrics]
        }
        return {**base_config, **config}