from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
import tensorflow as tf
from src.components.transformers.model.cnn_model import CNNModel
from src.components.transformers.model.encoder_block import EncoderBlock
from src.components.transformers.model.decoder_block import DecoderBlock
from src.components.transformers.model.image_caption_model import ImageCaptionModel
from src.components.transformers.model.lr_scheduler import LRSchedule
from src.components.transformers.transformers_config import TransformersConfig


@dataclass
class TrainConfig:
    transformers_config = TransformersConfig()


class TrainPipeline:
    def __init__(self,train_dataset,val_dataset):
        self.config = TrainConfig()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=15, restore_best_weights=True
        )
        self.EMBED_DIM = self.config.transformers_config.EMBED_DIM
        self.FF_DIM = self.config.transformers_config.FF_DIM
        self.saving_model = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.config.transformers_config.ARTIFACT_DIR,"chekcpoint_{epoch}.weights.h5"),
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            save_freq="epoch",
            initial_value_threshold=None,)
        self.model = None

    def build_model(self):
        cnn_model = CNNModel()
        encoder = EncoderBlock(
            embed_dim=self.EMBED_DIM, dense_dim=self.FF_DIM, num_heads=1
        )
        decoder = DecoderBlock(
            embed_dim=self.EMBED_DIM, ffnn_dim=self.FF_DIM, num_heads=2
        )
        caption_model = ImageCaptionModel(
            cnn_model = cnn_model.get_cnn_model(),
            encoder = encoder,
            decoder = decoder,
            image_aug = self.image_augmentation(),
        )
        return caption_model

    def image_augmentation(self):
        image_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.3),
        ])
        return image_augmentation
    
    def train_model(self):
        num_train_steps = len(self.train_dataset) * self.config.transformers_config.EPOCHS
        num_warmup_steps = num_train_steps // 15
        lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

        # Compile the model
        caption_model = self.build_model()
        caption_model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), loss=self.cross_entropy)

        # Fit the model
        history = caption_model.fit(
            self.train_dataset,
            epochs= self.config.transformers_config.EPOCHS,
            validation_data=self.val_dataset,
            callbacks=[self.saving_model],
        )
        self.model = caption_model
        