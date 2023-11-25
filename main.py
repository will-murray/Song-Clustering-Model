from music_space.music_space import music_space
from user_taste.user_taste import user_taste
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import pearsonr




MS = music_space('music_space/embeddings/npy/')
UT = user_taste('user_taste/data/')


def visualize_purity(k):
    MS.cluster_model(k)
    users = UT.get_all_users()
    num_groups_in = []
    for uid in users:
        memberships = [MS.label_dict[record[1]] for record in UT.get_listening_history(uid)]   
        unique_values, counts = np.unique(memberships, return_counts=True)
        num_groups_in.append(len(unique_values))

    plt.hist(x= num_groups_in,bins=k)
    plt.show()


class cluster_recommeder:

    def __init__(self,k):
        self.MS = music_space('music_space/embeddings/npy/')
        self.UT = user_taste('user_taste/data/')
        self.user_song_mat = self.init_user_item_matrix()
        self.clusters = KMeans(n_clusters= k,init='k-means++')
        self.user_label_dict = None
        

    def init_user_item_matrix(self):
        """Initialize the user-item matrix.
        Each row is a user and each column is a song (item)
        
        """

        users = self.UT.get_all_users()
        songs = self.MS.song_IDs

        user_song_matrix = np.zeros((users.shape[0], songs.shape[0]))
        sid_index_dict = {sid : index for index,sid in enumerate(self.MS.song_IDs)}

        
        for idx,uid in enumerate(users):
            user_likes = np.zeros(songs.shape[0])
            for record in UT.get_listening_history(uid):
                user_likes[sid_index_dict[record[1]]] = record[2]
            user_song_matrix[idx, :] = user_likes

        return user_song_matrix
    
    def fit(self):
        """This function initializes a dictionary that maps users
        to thier cluster label"""
        self.clusters.fit(self.user_song_mat)
        self.user_label_dict = {uid : label for uid,label in zip(self.UT.get_all_users(),self.clusters.labels_)}

    def recommend(self, uid):
        user_cluster_label = self.clusters.predict()

    


model = cluster_recommeder(5)
model.fit()
print(model.user_label_dict)
# model.recommend(model.UT.get_rand_user())
