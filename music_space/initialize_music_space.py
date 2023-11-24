import hdf5_getters as h5
import numpy as np
import os
import csv
from feature_extractor import *
from tqdm import tqdm


def list_song_files():
    root = "MillionSongSubset"
    files= []
    for prefix_1 in ['A', 'B']:
        child = root + '/' + prefix_1

        for prefix_2 in os.listdir(child):
            g_child = child + '/' + prefix_2
            
            for prefix_3 in os.listdir(g_child):
                gg_child = g_child + '/' + prefix_3
            
                for song_file_path in os.listdir(gg_child):
                    files.append(gg_child + '/' + song_file_path)

    return files



def get_all_artist_terms():
    terms = set()
    i = 0
    for song_path in tqdm(list_song_files()):
        song = h5.open_h5_file_read(song_path)
        for term in np.char.decode(h5.get_artist_terms(song)):
            terms.add(term)
        song.close()
        i+=1
    with open('embeddings/csv/term_list.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(terms)

    np.save(file = "embeddings/npy/term_list",arr=np.array(list(terms)))



def initialize_music_space(mode = 2):

    """
    1. Initialize and save the music space to music_space/embedding.
    Embedding is a numpy ndarray, saved as a .npy file
    The features in the music space are selected in the function grab song

    2. Initialize and save a vector of song_ID
    """

    processed = 0

    music_space = []
    song_IDs = []
    
    if mode == 2:
        all_terms = None
        with open('embeddings/term_frequency.csv') as file:
            reader = csv.reader(file)
            all_terms = [row[1] for row in reader]

        all_terms = np.array(all_terms)

    print("Processing songs:")
    for song_file_path in tqdm(list_song_files()):

        if mode == 1:
            song_ID, X = extractor_1(path = song_file_path)
        else:
            song_ID, X = extractor_2(path = song_file_path,TERMS_SET= all_terms)

        song_ID = song_ID.decode("UTF-8")
        music_space.append(X)
        song_IDs.append(song_ID)
        processed += 1

    if not os.path.exists( "embeddings" ):
        os.mkdir("embeddings")
    if not os.path.exists( "embeddings/csv" ):
        directory_path = os.path.join("embeddings", "csv")
        os.makedirs(directory_path) 
    if not os.path.exists( "embeddings/npy" ):
        directory_path = os.path.join("embeddings", "npy")
        os.makedirs(directory_path)

    csv_file = f'embeddings/csv/MSD_songs_{mode}.csv'
    csv_ids_file = f'embeddings/csv/MSD_IDs_{mode}.csv'
    npy_file = f'embeddings/npy/MSD_features_{mode}.npy'
    npy_ids_file = f'embeddings/npy/MSD_song_IDs_{mode}.npy'


    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in music_space:
            writer.writerow(row)
    with open(csv_ids_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for id in song_IDs:
                writer.writerow(id)

    np.save(file = os.getcwd() +'/'+ npy_file,arr = music_space)
    np.save(file = os.getcwd() +'/'+ npy_ids_file,arr = song_IDs)


if __name__=="__main__":
    initialize_music_space(mode=2)


