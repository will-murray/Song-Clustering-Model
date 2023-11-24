
import numpy as np
from sklearn.neighbors import KDTree

class USER():
    """
    Implementation of the user profile interface
    
    This class was implemented for efficient song score calulation.
    """

    def __init__(self,uid,ms,ut):

        self.user_history = ut.get_listening_history(uid)
        self.song_vectors = ms.feature_vectors[np.where(np.isin(ms.song_IDs, self.user_history[:,1])), : ][0]
        self.KD_TREE = KDTree(self.song_vectors)

    def get_song_score(self, song_vector):
        """
        return a value representing how much the user likes the recommended song (song_vector)
        """
        dist, index = self.KD_TREE.query(X = song_vector.reshape(1,-1), k = 1)
        

        score = int(self.user_history[int(index),2]) / np.linalg.norm(dist)
        return score
    


