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
        self.test_user_song_mat = self.init_user_item_matrix(test_set=True)

        self.clusters = KMeans(n_clusters= k,init='k-means++',n_init=30)
        
        self.user_label_dict = None
        self.__test_user_row_idx_dict = {uid:idx for idx,uid in enumerate(self.UT.get_all_users(test_set=True))}
        self.song_to_idx_dict = {sid:idx for idx,sid in enumerate(self.MS.song_IDs)}

    def init_user_item_matrix(self,test_set = False):
        """Initialize the user-item matrix.
        Each row is a user and each column is a song (item)
        
        """

        users = self.UT.get_all_users(test_set)
        songs = self.MS.song_IDs

        user_song_matrix = np.zeros((users.shape[0], songs.shape[0]))
        sid_index_dict = {sid : index for index,sid in enumerate(self.MS.song_IDs)}

        
        for idx,uid in enumerate(users):
            user_likes = np.zeros(songs.shape[0])
            for record in UT.get_listening_history(uid):
                user_likes[sid_index_dict[record[1]]] = record[2]
            user_song_matrix[idx, :] = user_likes

        return user_song_matrix
    
    def get_test_user_taste(self,uid):
        return self.test_user_song_mat[self.__test_user_row_idx_dict[uid],:]

    def fit(self):
        """This function initializes a dictionary that maps users
        to thier cluster label"""
        self.clusters.fit(self.user_song_mat)
        self.user_label_dict = {uid : label for uid,label in zip(self.UT.get_all_users(),self.clusters.labels_)}
        # self.estimate_unrated_centroid_values()

    def estimate_unrated_centroid_values(self):
        """Estimate values for songs which haven't been rated by any user in the cluster."""
        for cluster_label in range(self.clusters.n_clusters):
            cluster_center = self.clusters.cluster_centers_[cluster_label]

            # Find indices of zero values in the cluster center
            zero_indices = np.where(cluster_center == 0)[0]

            # If there are zero values in the cluster center
            if len(zero_indices) > 0:
                # Get the indices of non-zero values in the same cluster center
                non_zero_indices = np.where(cluster_center != 0)[0]

                # Estimate zero values based on non-zero values in the same cluster center
                for zero_idx in zero_indices:
                    # Simple average of non-zero values
                    estimated_value = np.mean(cluster_center[non_zero_indices])

                    # Update the cluster center with the estimated value
                    cluster_center[zero_idx] = estimated_value

    def users_in_cluster(self,cluster_idx):
        indices_in_cluster = [i for i, label in enumerate(self.clusters.labels_) if label == cluster_idx]
        return self.user_song_mat[indices_in_cluster, :]
    

    def recommend(self, uid, p):

        user_taste = self.get_test_user_taste(uid)

        # Find the centroids of all clusters
        cluster_centers = self.clusters.cluster_centers_

        #Find the pearson correlation between the user and each cluster center
        user_center_sim = [pearsonr(user_taste, center)[0] for center in cluster_centers]

        #Get the cluster indicies of the p clusters that are most similiar to the user
        p_similar_cluster_idx = sorted(range(len(user_center_sim)), key=lambda i: user_center_sim[i], reverse=True)[:p]

        
        weighted_avg = np.zeros(10000) 
        for idx in p_similar_cluster_idx:
            weighted_avg += self.clusters.cluster_centers_[idx]

        return weighted_avg/ p
            


model = cluster_recommeder(15)
model.fit()

for uid in model.UT.get_all_users(test_set=True):
    print("------------------------")
    yhat = model.recommend(uid, p =3)

    for record in model.UT.get_listening_history(uid,test_set = True):
        y = record[2]
        r = model.recommend(uid,p=3)
        song_idx = model.song_to_idx_dict[record[1]]
        yhat = r[song_idx]
        print(f"{y} : {yhat}")
