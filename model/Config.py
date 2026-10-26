import ml_collections

def brnn_config():
    model_config = ml_collections.ConfigDict()
    model_config.gru_units=200
    model_config.dense_units =  10
    model_config.drop_out=0.1
    model_config.l1= 0.1
    model_config.l2 = 0.1
    model_config.activation = 'tanh'
    model_config.recurrent_activation = 'relu'
    model_config.patience=20
    model_config.batch_size=16
    model_config.input_shape=(None,1)

    return model_config
def fl_config():
    config = ml_collections.ConfigDict()
    config.poolSize = 2000
    config.batch_size=16
    config.learning_rate=5e-5
    config.decay_steps=1000
    config.decay = 0.9
    config.risk_thread = 0.8
    config.check_keras_model = False
    config.check_gradients = False
    config.check_critical_threshold = False
    config.check_communication_rounds = True
    config.check_bound_of_maximum_timegap = False
    return config
