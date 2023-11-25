import numpy as np
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split

class music_space:

    def __init__(self, path):
        self.feature_vectors = np.load(path + "MSD_features_2.npy")
        self.song_IDs = np.load(path + "MSD_song_IDs_2.npy", allow_pickle=True)
    
    def vector_to_songID(self, song_vector):
        """returns the song_ID for the given song_vector"""

        distances = np.linalg.norm(self.feature_vectors - song_vector, axis=1)
        min_distance_index = np.argmin(distances)        
        song_id = self.song_IDs[min_distance_index]
        
        return song_id

    def songID_to_vector(self, song_id):
        """returns the song_vector for the given song_id"""

        index = np.where(self.song_IDs == song_id)[0]
        if index.size == 0:
            return None
        else:
            vector = self.feature_vectors[index[0]]
            return vector

    def cluster_model(self,k):
        self.model = KMeans(n_clusters=k, init='k-means++').fit(self.feature_vectors)
        self.label_dict = {sid: label for sid, label in zip(self.song_IDs, self.model.labels_)}

    def get_random_song(self):
        i = random.randint(0,self.feature_vectors.shape[0]-1)
        return self.song_IDs[i]


    
