from music_space.music_space import music_space
from user_taste.user_taste import user_taste
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import pearsonr






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


def kmeans_plus_prob_power(U, k, power=2):
    # Step 2: Select the initial centroid c1 to be up
    centroids = [U[np.random.choice(len(U))]]

    # Step 3: Repeat until k centroids are found
    while len(centroids) < k:
        # Step 4: Select the next centroid ci
        distances = np.linalg.norm(U - centroids[-1], axis=1)
        probabilities = distances**power / np.sum(distances**power)

        next_centroid = U[np.random.choice(len(U), p=probabilities)]

        # Append the next centroid to the list
        centroids.append(next_centroid)

    # Step 6: Return { c1, c2, · · · ck }
    return np.array(centroids)

class cluster_recommeder:

    def __init__(self,k,algo = 'kmeans'):
        assert algo in ['kmeans', "HAC"]
        self.MS = music_space('music_space/embeddings/npy/')
        self.UT = user_taste('user_taste/data/')

        self.user_song_mat = self.init_user_item_matrix()
        self.test_user_song_mat = self.init_user_item_matrix(test_set=True)

        self.__algo = algo
        self.__init_cluster_model(k)
        self.__fit()
        self.cluster_centers = self.__init_cluster_centers()
        
        self.user_label_dict = None
        self.__test_user_row_idx_dict = {uid:idx for idx,uid in enumerate(self.UT.get_all_users(test_set=True))}
        self.song_to_idx_dict = {sid:idx for idx,sid in enumerate(self.MS.song_IDs)}

    def init_user_item_matrix(self,test_set = False):
        """
        Initialize the user-song matrix.
        Each row is a user and each column is a song

        Value at row i, column j is the ith users rating
        of the jth song
        """
        if test_set:
            print("Initializing user-song matrix on test set..")
        else:
            print("Initializing user-song matrix on training set..")

        users = self.UT.get_all_users(test_set)
        songs = self.MS.song_IDs

        user_song_matrix = np.zeros((users.shape[0], songs.shape[0]))
        sid_index_dict = {sid : index for index,sid in enumerate(self.MS.song_IDs)}

        
        for idx,uid in enumerate(users):
            user_likes = np.zeros(songs.shape[0])
            for record in self.UT.get_listening_history(uid):
                user_likes[sid_index_dict[record[1]]] = record[2]
            user_song_matrix[idx, :] = user_likes

        return user_song_matrix

    def __init_cluster_model(self,k):
        if self.__algo == 'kmeans':
            self.clusters = KMeans(
                        n_clusters= k,
                        init= kmeans_plus_prob_power(self.user_song_mat, k),
                        n_init= 10
                        )
        else:
            self.clusters = AgglomerativeClustering(n_clusters=k,linkage='single')

    def __init_cluster_centers(self):
        print(f"saving {self.__algo} cluster centers...")
        if self.__algo == 'kmeans':
            self.cluster_centers = self.clusters.cluster_centers_
        else:
            # Assuming the model is already fit
            labels = self.clusters.labels_

            # Compute representative point for each cluster (e.g., centroid)
            cluster_centers = []
            unique_labels = set(labels)

            for i in range(1, len(unique_labels) + 1):
                print(f"label {i}")
                cluster_points = self.user_song_mat[labels == i]

                if cluster_points.shape[0] > 0:
                    center = cluster_points.mean(axis=0)
                    cluster_centers.append(center)
                else:
                    # Handle empty cluster
                    print(f"Warning: Cluster {i} is empty.")

            return np.array(cluster_centers)
    

    
    def get_test_user_taste(self,uid):
        return self.test_user_song_mat[self.__test_user_row_idx_dict[uid],:]

    def __fit(self):
        """This function initializes a dictionary that maps users
        to thier cluster label"""
        self.clusters.fit(self.user_song_mat)
        self.user_label_dict = {uid : label for uid,label in zip(self.UT.get_all_users(),self.clusters.labels_)}

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
    
    def cluster_distrubution(self):
        labels = self.clusters.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label + 1}: {count} points")

    def recommend(self, uid, p):

        user_taste = self.get_test_user_taste(uid)

        # Find the centroids of all clusters
        cluster_centers = self.cluster_centers

        #Find the pearson correlation between the user and each cluster center
        user_center_sim = [pearsonr(user_taste, center)[0] for center in cluster_centers]

        #Get the cluster indicies of the p clusters that are most similiar to the user
        p_similar_cluster_idx = sorted(range(len(user_center_sim)), key=lambda i: user_center_sim[i], reverse=True)[:p]

        
        weighted_avg = np.zeros(10000) 
        for idx in p_similar_cluster_idx:
            weighted_avg += self.clusters.cluster_centers_[idx]

        return weighted_avg/ p


def tune_k():

    k_vals = [40,50,60,100,200]
    errs = []
    for k in k_vals:
        print(f"k = {k}")
        model = cluster_recommeder(k)
        model.fit()
        model.cluster_distrubution()
        err = 0
        for uid in model.UT.get_all_users(test_set=True):
            yhat = model.recommend(uid, p =3)

            for record in model.UT.get_listening_history(uid,test_set = True):
                y = float(record[2])
                r = model.recommend(uid,p=3)
                song_idx = model.song_to_idx_dict[record[1]]
                yhat = r[song_idx]
                err += np.abs(y-yhat)

        errs.append(err / model.UT.get_all_users(test_set=True).shape[0])


    plt.plot(k_vals,errs)
    plt.show()
    print(f"errs = {errs}")

model = cluster_recommeder(5,'HAC')