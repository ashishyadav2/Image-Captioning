import os
from src.logger import logging
from src.exception import CustomException
import numpy as np
from tensorflow.keras.utils import to_categorical, plot_model, model_to_dot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils import load_object, save_model


class ModelTrainer:
    def __init__(self, data_transformation_config, train_set, test_set, model):
        self.max_caption_len = data_transformation_config.max_caption_len
        self.vocab_size = data_transformation_config.vocab_size
        self.image_names = data_transformation_config.image_ids
        self.tokenizer = data_transformation_config.tokenizer
        self.image_caption_map = load_object(
            data_transformation_config.image_caption_map
        )
        self.image_feature_map = load_object(
            data_transformation_config.image_feature_map
        )
        self.train_set = train_set
        self.test_set = test_set
        self.model = model
        self.file_path = data_transformation_config.DS_PATH
        self.model_path = os.path.join(self.file_path, "model.keras")

    def __data_generator(self, batch_size=32):
        X1, X2, y = list(), list(), list()
        counter = 0
        while True:
            for image_name in self.image_names:
                counter += 1
                captions = self.image_caption_map[image_name]

                for caption in captions:
                    sequence = self.tokenizer.texts_to_sequences([caption])[0]

                    for i in range(1, len(sequence)):
                        input_seq, output_seq = sequence[:i], sequence[i]
                        input_seq = pad_sequences(
                            [input_seq], maxlen=self.max_caption_len
                        )[0]
                        output_seq = to_categorical(
                            [output_seq], num_classes=self.vocab_size
                        )[0]

                        X1.append(self.image_feature_map[image_name][0])
                        X2.append(input_seq)
                        y.append(output_seq)

                if counter == batch_size:
                    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                    yield [X1, X2], y

                    X1, X2, y = list(), list(), list()
                    counter = 0

    def train_model(self, epochs=25):
        logging.info("model training started")
        batch_size = 32
        steps = len(self.train_set) // batch_size

        try:
            # for i in range(epochs):
            #     data_gen = self.__data_generator()
            #     self.model.fit(data_gen, epochs=1, steps_per_epoch=steps, verbose=0)
            logging.info("model training completed")
            save_model(self.model_path)

        except Exception as e:
            logging.info("error in model training")
            raise CustomException(e)

    def plot_model_structure(self):
        try:
            plot_model(
                self.model,
                to_file=os.path.join(self.file_path, "model_structure.png"),
                show_shapes=True,
                show_dtype=True,
                show_layer_activations=True,
            )
            logging.info("model plotted")

        except Exception as e:
            logging.info("error plotting model")
            raise CustomException(e)
