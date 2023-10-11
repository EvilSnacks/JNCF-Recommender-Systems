from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
import random, json, platform, os
# ------------------------------
RANDOM_SEED = 5717
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
plat = platform.platform()
gpu = len(tf.config.list_physical_devices('GPU')) > 0 # Checks to see if there is GPU on M1
tf.keras.mixed_precision.set_global_policy('float32') # Use 32ft instead 64 for faster
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2' # Use Accelerated Linear Algebra
os.environ['TF_METAL_DEVICE_PLACEMENT'] = 'metal' # Metal API is a low-level graphics API

import os
script_dir = os.path.dirname(os.path.abspath(__file__))[:-5]

# ------------------------------
from Model_TF import create_jncf_model, PreProcess_csv, Num_Users_Items
if __name__ == '__main__':
    train_users,\
        test_users,\
        train_items,\
        test_items,\
        train_ratings,\
        test_ratings,\
        train_extra_features,\
        test_extra_features = PreProcess_csv(os.path.join(script_dir,"dataset","processed",'User_Item.csv'))
    num_users, num_movies, num_features = Num_Users_Items(os.path.join(script_dir,"dataset","processed",'User_Item.csv'))

    with open(os.path.join(script_dir,"model",'hyperparameters.json'), 'r') as f:
        hyperparameters = json.load(f)
        print(json.dumps(hyperparameters, indent=4))
        print(f"Python Platform: {plat}")
        print(f"GPU is {'available' if gpu else 'not available'}")
        print('\n')

    jncf_model = create_jncf_model(M=int(num_movies),
                                   N=int(num_users),
                                   num_layers_df=int(hyperparameters['num_layers_df']),
                                   num_layers_di=int(hyperparameters['num_layers_di']),
                                   embedding_dim=int(hyperparameters['embedding_dim']),
                                   alpha=float(hyperparameters['alpha']),
                                   num_features=int(num_features),
                                   fusion_type=str(hyperparameters['fusion_type']),
                                   pairwise_loss_type=str(hyperparameters['pairwise_loss_type']),
                                   regularization=str(hyperparameters['regularization']),
                                   reg_rate=float(hyperparameters['reg_rate'])
                                   )
    jncf_model.summary()

    tensorboard = TensorBoard(
        log_dir=os.path.join(script_dir,"model","logs", "Model_Logs"),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=1,
        embeddings_freq=1,
    )

    jncf_model.fit(x=[train_users, train_items, train_extra_features],
                   y=train_ratings,
                   batch_size=int(hyperparameters['batch_size']),
                   epochs=int(hyperparameters['epochs']),
                   validation_data=([test_users, test_items, test_extra_features], test_ratings),
                   verbose=1,
                   callbacks=[tensorboard]
                   )

    jncf_model.save(os.path.join(script_dir,"model",'jncf_model.h5'))