import json, jsonlines, os
import pandas as pd
from tqdm import tqdm
json_len_users = 0
json_len_items = 0

script_dir = os.path.dirname(os.path.abspath(__file__))

def read_data():
    # Movielens Dataset 1 million

    # UserID::MovieID::Rating::Timestamp
    # no Timestamp
    ratings = pd.read_csv(os.path.join(script_dir, 'raw', 'ratings.dat'), sep='::', engine='python', header=None, usecols=[0, 1, 2],
                          names=['UserID', 'MovieID', 'Rating'])
    # ratings.dat example
    '''
    1::1193::5::978300760
    1::661::3::978302109
    1::914::3::978301968
    1::3408::4::978300275
    1::2355::5::978824291
    ...
    '''

    # UserID::Gender::Age::Occupation::Zip-code
    # no Zip-code
    users = pd.read_csv(os.path.join(script_dir, 'raw', 'users.dat'), sep='::', engine='python', header=None, usecols=[0, 1, 2, 3],
                        names=['UserID', 'Gender', 'Age', 'Occupation'])
    #users.dat example
    '''
    1::F::1::10::48067
    2::M::56::16::70072
    3::M::25::15::55117
    4::M::45::7::02460
    5::M::25::20::55455
    ...
    '''

    # MovieID::Title::Genres
    movies = pd.read_csv(os.path.join(script_dir, 'raw', 'movies.dat'), sep='::', engine='python', header=None, names=['MovieID', 'Title', 'Genres'],
                         encoding='latin-1')
    # movies.dat example
    '''
    1::Toy Story (1995)::Animation|Children's|Comedy
    2::Jumanji (1995)::Adventure|Children's|Fantasy
    3::Grumpier Old Men (1995)::Comedy|Romance
    4::Waiting to Exhale (1995)::Comedy|Drama
    5::Father of the Bride Part II (1995)::Comedy
    ...
    '''

    return ratings, users, movies


def prepare_mapping(movie_df):
    All_map = dict()
    gender_mapping = {
        0: "F",
        1: "M"
    }
    All_map["gender_mapping"] = gender_mapping

    occupation_mapping = {
        0: "other",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer"
    }
    All_map["occupation_mapping"] = occupation_mapping

    age_mapping = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+"
    }
    All_map["age_mapping"] = age_mapping

    genre_mapping = {
        0: "Action",
        1: "Adventure",
        2: "Animation",
        3: "Children",
        4: "Comedy",
        5: "Crime",
        6: "Documentary",
        7: "Drama",
        8: "Fantasy",
        9: "Film-Noir",
        10: "Horror",
        11: "Musical",
        12: "Mystery",
        13: "Romance",
        14: "Sci-Fi",
        15: "Thriller",
        16: "War",
        17: "Western"
    }
    All_map["genre_mapping"] = genre_mapping

    movie_df.set_index('MovieID', inplace=True)
    id_to_title = movie_df['Title'].to_dict()
    Final_map = All_map.copy()
    Final_map["MovieID_Title"] = id_to_title

    with open(os.path.join(script_dir, 'processed',"Mappings.json"), "w") as f:
        json.dump(Final_map, f, indent=4)

    del Final_map, id_to_title, movie_df, All_map

def user_movie_rating_collection(rat_pd, user_pd):
    with open(os.path.join(script_dir, 'processed', "TEMP_USER.json"), 'w') as f:
        unique_userids = rat_pd["UserID"].unique()
        i = 0
        for user_id in tqdm(unique_userids, desc='User to JSON', unit=" JSON Objects", smoothing=0.1):
            i += 1
            user_info = dict()
            user_info['ID'] = int(user_id)

            id_rating = rat_pd.loc[rat_pd['UserID'] == user_id, ['MovieID', 'Rating']]
            id_movie_rating = dict(zip(id_rating['MovieID'], id_rating['Rating']))
            user_info["User_Ratings"] = id_movie_rating

            gender_to_num = lambda x: 1 if x == 'M' else 0
            id_gender = user_pd.loc[user_pd['UserID'] == user_id]['Gender'].apply(gender_to_num).values[0]
            user_info["Gender"] = int(id_gender)

            id_age = user_pd.loc[user_pd['UserID'] == user_id, 'Age'].iloc[0]
            user_info["Age"] = int(id_age)

            id_occupation = user_pd.loc[user_pd['UserID'] == user_id, 'Occupation'].iloc[0]
            user_info["Occupation"] = int(id_occupation)

            json.dump(user_info, f)
            f.write('\n')

        global json_len_users
        json_len_users = i

def item_movie_rating_collection(rat_pd, mov_pd):
    l = len(rat_pd) + len(mov_pd["MovieID"].unique())
    global json_len_items
    json_len_items = l
    progress_bar = tqdm(total=l, desc='Movie to JSON', unit=" JSON Objects", smoothing=0.1)

    with open(os.path.join(script_dir, 'processed', "TEMP_ITEM.json"), "w") as f:
        map_name_id = {
            "Action": 0,
            "Adventure": 0,
            "Animation": 0,
            "Children": 0,
            "Comedy": 0,
            "Crime": 0,
            "Documentary": 0,
            "Drama": 0,
            "Fantasy": 0,
            "Film-Noir": 0,
            "Horror": 0,
            "Musical": 0,
            "Mystery": 0,
            "Romance": 0,
            "Sci-Fi": 0,
            "Thriller": 0,
            "War": 0,
            "Western": 0
        }

        movieid_usr_rat = dict()
        for idx, row in rat_pd.iterrows():
            movie_id = int(row["MovieID"])
            user_id = int(row["UserID"])
            rating = row["Rating"]

            if movie_id not in movieid_usr_rat:
                temp = {"UserIDs": {user_id: rating}}
                movieid_usr_rat[movie_id] = temp

            else:
                movieid_usr_rat[movie_id]["UserIDs"][user_id] = rating

            progress_bar.update(1)

        unique_movieids = mov_pd["MovieID"].unique()
        for movie_id in unique_movieids:
            movie_info = dict()
            movie_info["MovieID"] = int(movie_id)

            genre_mapping = map_name_id.copy()

            # Title
            id_title = str(mov_pd.loc[mov_pd['MovieID'] == movie_id, 'Title'].iloc[0])

            # Genres
            id_g = str(mov_pd.loc[mov_pd['MovieID'] == movie_id, 'Genres'].iloc[0])

            for category in genre_mapping.keys():
                if category in id_g:
                    genre_mapping[category] = 1

            movie_info["Title"] = id_title
            movie_info["Genres"] = genre_mapping

            # WARNING! Some MovieID do not have ratings (ex: 51)
            movie_id_rating = movieid_usr_rat.get(movie_id, {})
            if "UserIDs" in movie_id_rating:
                movie_info["UserID's_Rating"] = movie_id_rating["UserIDs"]
            else:
                movie_info["UserID's_Rating"] = {}

            json.dump(movie_info, f)
            f.write('\n')

            progress_bar.update(1)
    progress_bar.close()

def user_preprocessing():
    processed_user_ratings = pd.DataFrame(columns=['UserID', 'MovieID', 'Rating', 'Gender', 'Age', 'Occupation'])
    progress_bar = tqdm(total=json_len_users, desc='USER_JSON to CSV', unit=" JSON Objects", smoothing=0.1)

    with open(os.path.join(script_dir, 'processed', "TEMP_USER.json"), "r") as f:
        for line in f:
            json_data = json.loads(line)

            user_ratings_df = pd.DataFrame(list(json_data['User_Ratings'].items()), columns=['MovieID', 'Rating'])

            user_ratings_df['UserID'] = json_data['ID']
            user_ratings_df['Gender'] = json_data['Gender']
            user_ratings_df['Age'] = json_data['Age']
            user_ratings_df['Occupation'] = json_data['Occupation']

            user_ratings_df = user_ratings_df[['UserID', 'MovieID', 'Rating', 'Gender', 'Age', 'Occupation']]

            processed_user_ratings = pd.concat([processed_user_ratings, user_ratings_df], ignore_index=True)
            progress_bar.update(1)

    progress_bar.close()
    return processed_user_ratings

def item_preprocessing():
    columns = ['MovieID', 'UserID', 'Rating', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
               'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
               'Thriller', 'War', 'Western']
    batch_data = []
    batch_size = 1000
    processed_movie_ratings = pd.DataFrame(columns=columns)

    with jsonlines.open(os.path.join(script_dir, 'processed',"TEMP_ITEM.json"), "r") as f:
        movie_data = list(f)

    progress_bar = tqdm(total=len(movie_data), desc='MOVIE_JSON to CSV', unit=" JSON Objects", smoothing=0.1)

    for json_data in movie_data:
        movie_id = json_data['MovieID']
        user_ratings = json_data["UserID's_Rating"]
        genres = json_data['Genres']

        for user_id, rating in user_ratings.items():
            row_data = {'MovieID': movie_id, 'UserID': user_id, 'Rating': rating}
            row_data.update(genres)
            batch_data.append(row_data)

            if len(batch_data) == batch_size:
                batch_df = pd.DataFrame(batch_data, columns=columns)
                processed_movie_ratings = pd.concat([processed_movie_ratings, batch_df], ignore_index=True)
                batch_data = []

        progress_bar.update(1)

    if batch_data:
        batch_df = pd.DataFrame(batch_data, columns=columns)
        processed_movie_ratings = pd.concat([processed_movie_ratings, batch_df], ignore_index=True)

    progress_bar.close()
    return processed_movie_ratings

def combine_user_item_data(temp_user_path, temp_item_path):
    with jsonlines.open(temp_user_path) as user_file:
        user_data = list(user_file)
    with jsonlines.open(temp_item_path) as item_file:
        item_data = list(item_file)

    combined_data = {"UserID": [], "MovieID": [], "Rating": [], "Gender": [], "Age": [], "Occupation": []}
    for genre in item_data[0]["Genres"]:
        combined_data[genre] = []

    progress_bar = tqdm(total=len(user_data), desc='Combined Data', unit='users/movies')
    for user in user_data:
        user_id = user["ID"]
        user_ratings = user["User_Ratings"]
        user_gender = user["Gender"]
        user_age = user["Age"]
        user_occupation = user["Occupation"]

        for movie in item_data:
            movie_id = movie["MovieID"]
            movie_ratings = movie["UserID's_Rating"]

            if str(movie_id) in user_ratings and str(user_id) in movie_ratings:
                combined_data["UserID"].append(user_id)
                combined_data["MovieID"].append(movie_id)
                combined_data["Rating"].append(user_ratings[str(movie_id)])
                combined_data["Gender"].append(user_gender)
                combined_data["Age"].append(user_age)
                combined_data["Occupation"].append(user_occupation)

                for genre in item_data[0]["Genres"]:
                    combined_data[genre].append(movie["Genres"][genre])

        progress_bar.update(1)
    progress_bar.close()

    combined_data_df = pd.DataFrame(combined_data)

    return combined_data_df

if __name__ == '__main__':
    #(0) Context
    '''
    Make sure movies.dat, ratings.dat, & users.dat are in the same directory as this python file
    This python file will generate multiple files, the main one being User_Item.csv
    '''

    # (1) Raw_Data
    RAW_rating, RAW_user, RAW_movie = read_data()
    RAW_rating['Rating'] = RAW_rating['Rating'].apply(lambda x: round(x / 5, 10))
    prepare_mapping(movie_df=RAW_movie.copy())

    # (2.2) Create JSON file with JSON objects of all useful information needed to create user-item & item-user
    # "matrix" (input vector)
    user_movie_rating_collection(RAW_rating, RAW_user)
    item_movie_rating_collection(RAW_rating, RAW_movie)
    del RAW_rating, RAW_user, RAW_movie

    # (3) Pre-Processing
    user_processed_pd = user_preprocessing()
    user_processed_pd.to_csv(os.path.join(script_dir, 'processed', 'RAW_user_input.csv'))
    item_processed_pd = item_preprocessing()
    item_processed_pd.to_csv(os.path.join(script_dir, 'processed',"RAW_item_input.csv"))

    # (4) Saving final file
    combined_data_df = combine_user_item_data(os.path.join(script_dir, 'processed', "TEMP_USER.json"), os.path.join(script_dir, 'processed',"TEMP_ITEM.json"))
    combined_data_df.to_csv(os.path.join(script_dir, 'processed','User_Item.csv'), encoding='UTF-8')

