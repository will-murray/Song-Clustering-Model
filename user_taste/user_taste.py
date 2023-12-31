#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Implementation of user taste module

NOTE: This module requires the user taste dataset to be available. This file can be generated by running entp_reader.py


"""
import user_taste as UT
import numpy as np 
from numpy.random import randint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ^ commented out these modules, cvxpy doesn't like them

class user_taste():

    def __init__(self,path):
        self.taste_space, self.test_set = train_test_split( np.load(path + 'user_taste.npy')[:10000] , test_size=0.2)
        self.taste_dictionary = self.__init_taste_dictionary()
        self.score_matrix = None
        """
        score matrix format:
            n = number of users
            m = number of songs
            the matrix must be n x m:
            the value at row i, column j is the ith users rating of the jth song
        """



    def __init_taste_dictionary(self):
        """constructs a dictionary of (sid,uid) : score pairs"""
        #group the sid and uid together into tuples
        keys = map(tuple, self.taste_space[:, :2])

        #cast the scores to int
        values = self.taste_space[:, 2].astype(int)

        taste_dictionary = dict(zip(keys, values))
        return taste_dictionary

    def get_song_score(self,uid,sid):
        """return the score given a user id (uid) and song id (sid)
        Dont use this function
        """

        listening_history = self.get_listening_history(uid)
        for song in list(listening_history[:, 1]):
            if song == sid:
                return self.taste_dictionary[(uid,sid)]

        return 0
    
    def get_rand_user(self, test_set = False):
        """returns the user id (uid) of a random user in the user taste dataframe"""

        if test_set:
            i = randint(0,self.test_set.shape[0]-1)
            return self.test_set[i,0]
        
        i = randint(0,self.taste_space.shape[0]-1)
        return self.taste_space[i,0]
    

    
    def get_listening_history(self,uid,test_set = False):
        """return all the songs which a given user has listened to"""
        if test_set:
            uid_records = self.test_set[:,0] == uid
            return self.test_set[uid_records]
        
        uid_records = self.taste_space[:,0] == uid
        return self.taste_space[uid_records]

    def get_all_users(self,test_set = False):
        if test_set:
            return np.unique(self.test_set[:,0])

        return np.unique(self.taste_space[:,0])

    def get_all_songs(self):
        return np.unique(self.taste_space[:,1])



class MC_score_matrix(user_taste):
    
    '''
    extension of user_taste to include  implementation of score matrix
    and a key-index mapping
    '''
    
    def __init__(self,path,solve=False):
        super().__init__(path)
        self.index_dictionary = self.__get_index_dictionary()
        
        if solve:
            self.__init_score_matrix(path)
            self.score_matrix = self.__MC_solve(path)
        
        else:
            self.score_matrix = np.load(path+'completed_score_matrix.npy')
        
        
    def get_song_score(self,key):
        '''get song score from matrix'''
        
        index = self.index_dictionary(key)
        return self.score_matrix(index)

    def __init_score_matrix(self,path):
        '''initialize score matrix with given song scores from taste space'''
            
            
        # matrix will be indexed lexographically, rowwise by user, columwise by song
        user_count = len(self.get_all_users())
        song_count = len(self.get_all_songs())    
    
        self.score_matrix = np.zeros((user_count,song_count))
          
        for (key,index) in self.index_dictionary.items():
            self.__add_score(key, index)
        
        
        
        
    
    
    def __add_score(self,key,index):
        if key in self.taste_dictionary:
            self.score_matrix[index] = self.taste_dictionary[key]
            
            
        
    def __get_index_dictionary(self):
        '''
        create a dictionary that maps taste_dictionary (uid,sid) key values to score matrix indices (i,j)
        representing the rating of the jth song (sid) by the ith user (uid)
        '''
        users = sorted(self.get_all_users())
        songs = sorted(self.get_all_songs())
        
        indices = [(i,j) for i in range(len(users)) for j in range(len(songs))]
        keys = [(user,song) for user in users for song in songs]
        
        
        index_dictionary = dict(zip(keys,indices)) 
        return index_dictionary
    

    
    
    def __MC_solve(self,path):
        import cvxpy as cp
        
        '''set-up and solve optimization problem and constraints for CVX'''
        n,m = self.score_matrix.shape
        
        X = cp.Variable((n,m))
        objective = cp.Minimize(cp.atoms.normNuc(X))
        constraints = []
        
        # get nonzero matrix entries 
        I,J = np.where(self.score_matrix!=0)
        nonzero_entries = [(i,j) for (i,j) in zip(I,J)]
        
        # define constraints to optimization problem
        print('Setting constraints...')
        for i in range(n):
            for j in range(m):
                if (i,j) in nonzero_entries:
                    C = X[i,j] == self.score_matrix[i,j]
                    constraints.append(C)
                else:
                    # constraining to positive scores only. Not necessary to obtain solution
                    C = X[i,j] >= 0
                    constraints.append(C)
        
        
        problem = cp.Problem(objective,constraints)
        
        print('Solving Matrix Completion Optimization problem...')
        # start = timeit.default_timer()
        problem.solve()  # this takes a while (~53 mins)
        # end = timeit.default_timer()
        
        print('Finished')
        print(f'Runtime: {end-start}')
        print(f'Status: {problem.status}')
        print(f'Objective: {problem.value}')
        
        np.save(file=path+'completed_score_matrix.npy',arr = X.value) 
        
        return X.value
        
        
        