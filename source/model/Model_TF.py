# M1 TensorFlow
# https://developer.apple.com/metal/tensorflow-plugin/
from tensorflow.keras.layers import Dense, Concatenate, Multiply, Input, Embedding, Input, Flatten, Dropout
from tensorflow.keras.models import Model
import pandas as pd
from tensorflow.keras import backend as K
from keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#New
def jncf_loss(alpha, pairwise_loss_type):
    def loss(y_true, y_pred):
        # Calculate L_pairwise: Pair-wise loss
        y_pred_pos = y_pred[:, 0]  # positive items
        y_pred_neg = y_pred[:, 1:]  # negative items
        pairwise_diff = y_pred_pos[:, None] - y_pred_neg

        if pairwise_loss_type == 'bpr_max':
            l_pairwise = - K.mean(K.log(K.sigmoid(pairwise_diff)))
        elif pairwise_loss_type == 'top1':
            l_pairwise = K.mean(K.sigmoid(pairwise_diff) + K.sigmoid(pairwise_diff**2))
        '''
        elif pairwise_loss_type == 'top1_max':
            l_pairwise = K.mean(K.sigmoid(pairwise_diff) - K.square(y_pred_pos))
        '''

        # L_cross_entropy
        y_true_normalized = y_true[:, 0] / K.max(y_true[:, 0])  # normalize y_true
        l_cross_entropy = - K.mean(
            y_true_normalized * K.log(y_pred_pos) + (1 - y_true_normalized) * K.log(1 - y_pred_pos))

        # Hybrid loss
        hybrid_loss = alpha * l_pairwise + (1 - alpha) * l_cross_entropy
        return hybrid_loss
    return loss

def create_jncf_model(M, N, num_layers_df, num_layers_di, embedding_dim, alpha, num_features, fusion_type, pairwise_loss_type, regularization, reg_rate):
    # Regularizer
    if regularization == 'L1': regularizer = l1(reg_rate)
    elif regularization == 'L2': regularizer = l2(reg_rate)
    else: regularizer = None
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    extra_features_input = Input(shape=(num_features,), name='extra_features_input')
    # Embedding layers
    user_embedding = Embedding(input_dim=N, output_dim=embedding_dim)(user_input)
    user_net = Flatten()(user_embedding)
    item_embedding = Embedding(input_dim=M, output_dim=embedding_dim)(item_input)
    item_net = Flatten()(item_embedding)
    # DF network for users
    for _ in range(num_layers_df - 1):
        user_net = Dense(embedding_dim, activation='relu')(user_net)
        if regularization == 'Dropout':
            user_net = Dropout(reg_rate)(user_net)
    # DF network for items
    for _ in range(num_layers_df - 1):
        item_net = Dense(embedding_dim, activation='relu')(item_net)
        if regularization == 'Dropout':
            item_net = Dropout(reg_rate)(item_net)
    # Fusion of user, item, and extra features
    if fusion_type == 'concat': fused_features = Concatenate()([user_net, item_net, extra_features_input])
    elif fusion_type == 'multiply':
        extra_features_net = Dense(embedding_dim)(extra_features_input)
        fused_features = Multiply()([user_net, item_net, extra_features_net])
    # DI network
    interaction_net = fused_features
    for _ in range(num_layers_di): interaction_net = Dense(embedding_dim, activation='relu', kernel_regularizer=regularizer)(interaction_net)
    # Output layer
    output = Dense(1, activation='sigmoid')(interaction_net)
    model = Model(inputs=[user_input, item_input, extra_features_input], outputs=output)
    model.compile(loss=jncf_loss(alpha, pairwise_loss_type=pairwise_loss_type), optimizer='adam', metrics=['mae', 'mse'])
    return model

def PreProcess_csv(fname: str):
    data = pd.read_csv(fname, index_col=0)
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)

    # Normalize
    data['Rating'] = data['Rating'] / 5

    # Scale the extra features
    scaler = StandardScaler()
    data.iloc[:, 4:] = scaler.fit_transform(data.iloc[:, 4:])

    users = data['UserID'].values
    items = data['MovieID'].values
    ratings = data['Rating'].values
    extra_features = data.drop(columns=['UserID', 'MovieID', 'Rating'], axis=1).values

    # Sampling based on UserID
    train_users,test_users, train_items, test_items, train_ratings, test_ratings, train_extra_features, test_extra_features = train_test_split(
        users, items, ratings, extra_features, test_size=0.2, random_state=42, stratify=users)

    return train_users, test_users, train_items, test_items, train_ratings, test_ratings, train_extra_features, test_extra_features

def PreProcess_graphs(fname: str):
    data = pd.read_csv(fname, index_col=0)
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)

    # Normalize ratings
    #data['Rating'] = data['Rating'] / 5

    # Scale extra features
    scaler = StandardScaler()
    data.iloc[:, 4:] = scaler.fit_transform(data.iloc[:, 4:])

    users = data['UserID'].values
    items = data['MovieID'].values
    ratings = data['Rating'].values
    extra_features = data.drop(columns=['UserID', 'MovieID', 'Rating'], axis=1).values

    # Sampling based on UserID
    train_users,test_users, train_items, test_items, train_ratings, test_ratings, train_extra_features, test_extra_features = train_test_split(
        users, items, ratings, extra_features, test_size=0.2, random_state=42, stratify=users)

    return train_users, test_users, train_items, test_items, train_ratings, test_ratings, train_extra_features, test_extra_features

def Num_Users_Items(fname: str):
    data = pd.read_csv(fname, index_col=0)
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)

    num_users = data['UserID'].nunique()
    num_movies = data['MovieID'].nunique()
    data = data.drop(['UserID', 'MovieID', 'Rating'], axis=1)

    num_features = len(data.columns)

    return num_users, num_movies, num_features


if __name__ == '__main__':
    pass



