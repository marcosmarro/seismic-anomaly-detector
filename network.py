import keras
from keras import layers, models

class CNN(keras.Model):
    def __init__(self, input_shape):
        super().__init__()

        # ----- Encoder -----
        self.encoder = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(32, kernel_size=9, padding='same', activation='relu'),
            layers.MaxPooling1D(pool_size=2),      # 6000 → 3000
            layers.Conv1D(64, kernel_size=9, padding='same', activation='relu'),
            layers.MaxPooling1D(pool_size=2),      # 3000 → 1500
            layers.Conv1D(128, kernel_size=9, padding='same', activation='relu'),
            layers.MaxPooling1D(pool_size=2),      # 1500 → 750
        ])

        # latent representation
        self.latent = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')

        # ----- Decoder -----
        self.decoder = models.Sequential([
            layers.UpSampling1D(size=2),           # 750 → 1500
            layers.Conv1D(128, kernel_size=9, padding='same', activation='relu'),
            layers.UpSampling1D(size=2),           # 1500 → 3000
            layers.Conv1D(64, kernel_size=9, padding='same', activation='relu'),
            layers.UpSampling1D(size=2),           # 3000 → 6000
            layers.Conv1D(32, kernel_size=9, padding='same', activation='relu'),
            layers.Conv1D(filters=input_shape[1], kernel_size=9, padding='same', activation='linear')
        ])

    def call(self, x):
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        return x