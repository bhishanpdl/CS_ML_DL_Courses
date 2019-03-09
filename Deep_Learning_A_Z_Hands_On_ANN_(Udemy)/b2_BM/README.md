# Creating rating for each user for each movies
```python
# Numpy method is faster
# Faster method: 827 µs
import numpy as np
import pandas as pd

# user 1 is missing here, user 4 is 4th user.
df = pd.DataFrame({'user': [2,2,2,4,4,4],
                 'movie': [1,2,5,2,4,6],
                 'rating': [2,4,4,2,8,3]})

def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

nb_users = 5
nb_movies = 7

training_set = df[['user','movie','rating']].values
convert(training_set)

*** Result***
[[ 0.  0.  0.  0.  0.  0.  0.]
 [ 2.  4.  0.  0.  4.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  2.  0.  8.  0.  3.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]]


# Pandas is slower
# Slower method: 1.99 ms

import numpy as np
import pandas as pd

# user 1 is missing here, user 4 is 4th user.
df = pd.DataFrame({'user': [2,2,2,4,4,4],
                 'movie': [1,2,5,2,4,6],
                 'rating': [2,4,4,2,8,3]})



def ratings_4users(df, n_users, n_movies):
    movie = np.array(df.groupby('user')['movie'].apply(lambda x: x.tolist()).tolist())
    rating = np.array(df.groupby('user')['rating'].apply(lambda x: x.tolist()).tolist())
    rating_lst = np.zeros((n_users,n_movies), dtype=float)

    for i,u in enumerate(df['user'].unique()):
        rating_lst[u-1][ np.array(movie[i]) -1] = rating[i]

    return rating_lst

n_users = 5
n_movies = 7
ratings_4users(df, n_users, n_movies)
```
