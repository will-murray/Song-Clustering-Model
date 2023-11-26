from user_taste.user_taste import user_taste
import matplotlib.pyplot as plt
import numpy as np

def play_count_dist():
    UT = user_taste('user_taste/data/')
    X = list(UT.taste_space[:, 2].astype('int'))
    plt.xlim(0,257)
    plt.hist(X, bins = 50)
    plt.yscale('log')
    plt.title("Distrubution of play counts in ENTP")
    print(max(X))
    plt.show()


def num_songs_per_user():
    UT = user_taste('user_taste/data/')
    users, counts = np.unique(UT.taste_space[:,0],return_counts=True)
    

    plt.hist(counts, bins = 20)
    plt.yscale('log')
    plt.title("Songs per User distrubution amoung the first 15 million ENTP records")
    plt.show()

num_songs_per_user()