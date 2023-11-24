#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm



def process(entp_file):
    """
        This function extracts the raw song id, user id, score triplets.
            - Only the triplets whose song id is found in the million song subset are kept
            - the resulting dataset is stored at user_taste/data/user_taste.npy

        
    """
    # retrieve unique user and song id's
    user_taste = []
    scores = []

    MSD_song_ids = np.load('../music_space/embeddings/npy/MSD_song_IDs_2.npy')
    MSD_song_ids = MSD_song_ids.astype(str)  # Convert MSD_song_ids to string type

    i = 1
    with open(entp_file, 'r') as input_file:

        for line in tqdm(input_file, total=44000000):

            uid, sid, score = line.strip().split()
            user_taste.append([uid, sid, score])
            scores.append(score)

            i+=1
            # #this was implemented because creating the whole dataset will 2 hours or so...
            # if i == num_lines_to_proc:
            #     break
    

    
    user_taste = np.array(user_taste)
    print(f"Total Ratings = {user_taste.shape[0]}")

    user_taste = user_taste[np.isin(user_taste[:, 1], MSD_song_ids)]
    print(f"Ratings in MSD = {user_taste.shape[0]}")

    unique_user_ids, user_id_counts = np.unique(user_taste[:, 0], return_counts=True)
    user_ids_at_least_10 = unique_user_ids[user_id_counts >= 10]
    user_taste = user_taste[np.isin(user_taste[:, 0], user_ids_at_least_10)]
    print(f"Filter for users with 10+ ratings = {user_taste.shape[0]}")


    np.save(file = 'data/user_taste.npy',arr = user_taste)

def sort_by_num_ratings():
    # get a sorted array of user id's and rating counts
    
    user_taste = np.load('data/user_taste.npy')
    uid = user_taste[:,:1]
    uid,counts = np.unique(uid,return_counts=True) # get unique uids and count of ratings

    sorted_inds = np.argsort(counts)[::-1] #sort by decreasing count 
     
    counts = counts[sorted_inds].astype(str)
    uid = uid[sorted_inds]
        
    rating_counts = np.stack((uid,counts),axis=1)   
    np.save(file='data/rating_counts.npy',arr = rating_counts)   
        
    


if __name__=="__main__":
    process("data/train_triplets.txt")
    sort_by_num_ratings()