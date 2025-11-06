
import ml_collections

def brnn_config():
    model_config = ml_collections.ConfigDict()
    model_config.gru_units=200
    model_config.dense_units =  10
    model_config.tol = 1e-5
    model_config.drop_out=0.1
    model_config.l1= 0.2
    model_config.l2 = 0.2
    model_config.activation = 'tanh'
    model_config.recurrent_activation = 'relu'
    model_config.patience=5
    model_config.batch_size=128
    model_config.input_shape=(None,1)

    return model_config
def fl_config():
    config = ml_collections.ConfigDict()
    config.poolSize = 200
    config.trainSize = 100
    config.testSize = 50
    config.predictSize = 5
    config.NUM_ROUNDS =80
    config.batch_size=128
    config.learning_rate=1e-4
    return config
