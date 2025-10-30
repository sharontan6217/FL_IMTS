
import ml_collections

def brnn_config():
    config = ml_collections.ConfigDict()
    config.gru_units=200
    config.dense_units = 10
    config.tol = 1e-5
    config.drop_out=0.1
    config.l1= 0.2
    config.l2 = 0.2
    config.activation='tanh'
    config.recurrent_activation='relu'
    config.patience=5
    config.batch_size=128
    config.input_shape=(None,1)

    return config
def fl_config():
    config = ml_collections.ConfigDict()
    config.poolSize = 700
    config.trainSize = 500
    config.testSize = 100
    config.predictSize = 10
    config.NUM_ROUNDS =8
    config.batch_size=128
    config.learning_rate=5e-5
    return config
