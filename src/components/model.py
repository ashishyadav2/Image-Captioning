from src.logger import logging
from src.exception import CustomException
from tensorflow.keras import models, layers


class Model:
    def __init__(self, max_caption_len, vocab_size):
        self.max_caption_len = max_caption_len
        self.vocab_size = vocab_size

    def get_model(self):
        try:
            input1 = layers.Input(shape=(4096,))
            dropout_layer_fe = layers.Dropout(0.5)(input1)
            fc3 = layers.Dense(units=256, activation="relu")(dropout_layer_fe)

            input2 = layers.Input(shape=(self.max_caption_len,))
            embedding_layer = layers.Embedding(
                input_dim=self.vocab_size, output_dim=256, mask_zero=True
            )(input2)
            dropout_layer_en = layers.Dropout(0.5)(embedding_layer)
            lstm_layer = layers.LSTM(256)(dropout_layer_en)

            fc3 = layers.add([fc3, lstm_layer])
            fc4 = layers.Dense(units=256, activation="relu")(fc3)
            fc5 = layers.Dense(units=self.vocab_size, activation="softmax")(fc4)

            model = models.Model(inputs=[input1, input2], outputs=fc5)
            model.compile(loss="categorical_crossentropy", optimizer="adam")

            logging.info("model created")

        except Exception as e:
            logging.info("error creating model")
            raise CustomException(e)

        return model
