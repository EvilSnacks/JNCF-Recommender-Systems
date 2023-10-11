import numpy as np
import tensorflow as tf
import random, json, platform, os
# ----------------------------------------------------
RANDOM_SEED = 5717
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
plat = platform.platform()
gpu = len(tf.config.list_physical_devices('GPU')) > 0 #checks to see if there is GPU on M1
tf.keras.mixed_precision.set_global_policy('float32') #Use 32ft instead 64 for faster
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2' #Use Accelerated Linear Algebra
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import os
script_dir = os.path.dirname(os.path.abspath(__file__))[:-5]
# ----------------------------------------------------
from Model_TF import PreProcess_csv,PreProcess_graphs, Num_Users_Items, jncf_loss, create_jncf_model

def visualize_model_performance(test_users, test_items, test_extra_features, test_ratings, batch_size=100000):
    model = tf.keras.models.load_model(os.path.join(script_dir,"model",'jncf_model.h5'), compile=False)
    model.compile(optimizer='adam', loss=jncf_loss(alpha=float(hyperparameters['alpha']),
                                                    pairwise_loss_type=str(hyperparameters['pairwise_loss_type'])
                                                    )
                  )

    def predict_in_batches(users, items, extra_features, model):
        num_samples = len(users)
        num_batches = math.ceil(num_samples / batch_size)
        predictions = np.zeros(num_samples)
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            batch_predictions = model.predict([users[start:end], items[start:end], extra_features[start:end]]).flatten()
            predictions[start:end] = batch_predictions
        return predictions

    original_predictions = predict_in_batches(test_users, test_items, test_extra_features, model)

    average_difference = []
    genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    attributes = ['Age', 'Gender', 'Occupation'] + genres
    for i, attr in enumerate(attributes):
        original_column = test_extra_features[:, i].copy()
        perturbed_column = original_column + np.random.normal(0, 0.01, size=len(original_column))
        perturbed_extra_features = test_extra_features.copy()
        perturbed_extra_features[:, i] = perturbed_column

        perturbed_predictions = predict_in_batches(test_users, test_items, perturbed_extra_features, model)
        diff = np.abs(perturbed_predictions - original_predictions)
        avg_diff = np.mean(diff)
        average_difference.append(avg_diff)

    average_difference = pd.Series(average_difference, index=attributes)
    sorted_difference = average_difference.sort_values()

    fig, ax = plt.subplots()
    sorted_difference.plot(kind='bar', ax=ax)
    ax.set_title('Average difference in predictions by attribute')
    ax.set_xlabel('Attribute')
    ax.set_ylabel('Average difference')

    for i, v in enumerate(sorted_difference):
        ax.text(i, v + 0.001, '{:.4f}'.format(v), ha='center', va='bottom', fontweight='bold')
    plt.show()

def visualize_on_MovieID(movie_id, test_users, test_items, test_ratings, predictions):
    movie_indices = [i for i, item in enumerate(test_items) if item == movie_id]

    users_for_movie = [test_users[i] for i in movie_indices]
    true_ratings_for_movie = [test_ratings[i] for i in movie_indices]
    predicted_ratings_for_movie = [predictions[i] for i in movie_indices]

    sorted_indices = sorted(range(len(users_for_movie)), key=lambda k: users_for_movie[k])
    sorted_users = [users_for_movie[i] for i in sorted_indices]
    sorted_true_ratings = [true_ratings_for_movie[i] for i in sorted_indices]
    sorted_predicted_ratings = [predicted_ratings_for_movie[i] for i in sorted_indices]

    plt.plot(sorted_users, sorted_true_ratings, label="True Ratings", marker='o')
    plt.plot(sorted_users, sorted_predicted_ratings, label="Predicted Ratings", marker='x')

    plt.xlabel("User ID")
    plt.ylabel("Rating")
    plt.title(f"True and Predicted Ratings for Movie {movie_id}")
    plt.legend()
    plt.show()

def visualize_on_all_movies(test_users, test_items, test_ratings, predictions):
    movie_errors = {}
    abs_error_sum = 0.0
    rating_sum = 0.0
    for i in range(len(test_users)):
        movie_id = test_items[i]
        true_rating = test_ratings[i]
        predicted_rating = predictions[i]
        abs_error = abs(true_rating - predicted_rating)
        error = true_rating - predicted_rating

        abs_error_sum += abs_error
        rating_sum += true_rating

        if movie_id not in movie_errors:
            movie_errors[movie_id] = {'error_sum': error, 'count': 1}
        else:
            movie_errors[movie_id]['error_sum'] += error
            movie_errors[movie_id]['count'] += 1

    avg_errors = {movie_id: error_data['error_sum'] / error_data['count'] for movie_id, error_data in movie_errors.items()}

    sorted_movie_ids = sorted(avg_errors.keys())
    sorted_avg_errors = [avg_errors[movie_id] for movie_id in sorted_movie_ids]

    overall_abs_error = abs_error_sum / len(test_users)
    overall_rating = rating_sum / len(test_users)
    overall_percent_error = overall_abs_error / overall_rating

    plt.axhline(y=overall_percent_error, color='r', linestyle='--', label='Overall Percent Error')

    plt.plot(sorted_movie_ids, sorted_avg_errors, label="Average Error", marker='o')

    plt.xlabel("Movie ID")
    plt.ylabel("Average Error")
    plt.title("Average Error for All Movies")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_users, test_users, train_items, test_items, train_ratings, test_ratings, train_extra_features, test_extra_features = PreProcess_graphs(os.path.join(script_dir,"dataset","processed",'User_Item.csv'))
    num_users, num_movies, num_features = Num_Users_Items(os.path.join(script_dir,"dataset","processed",'User_Item.csv'))

    with open(os.path.join(script_dir,"model",'hyperparameters.json'), 'r') as f:
        hyperparameters = json.load(f)
        print(hyperparameters)

    # Load model
    loaded_model = tf.keras.models.load_model(os.path.join(script_dir,"model",'jncf_model.h5'), compile=False)
    loaded_model.compile(optimizer='adam', loss=jncf_loss(alpha=float(hyperparameters['alpha']),
                                                          pairwise_loss_type=str(hyperparameters['pairwise_loss_type'])
                                                          )
                         )
    predictions = loaded_model.predict([test_users, test_items, test_extra_features])

    #Attributes
    visualize_model_performance(test_users, test_items, test_extra_features, test_ratings)

    # Call the visualize function with a specific movie ID
    movie_id = 5
    visualize_on_MovieID(movie_id, test_users, test_items, test_ratings, predictions)

    # Call the visualize_on_all_movies function
    visualize_on_all_movies(test_users, test_items, test_ratings, predictions)
