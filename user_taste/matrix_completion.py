#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implement Exact Matrix Completion to estimate empty entries of score matrix in user_taste module
"""

import numpy as np
from user_taste import user_taste
import timeit

    

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
        start = timeit.default_timer()
        problem.solve()  # this takes a while (~53 mins)
        end = timeit.default_timer()
        
        print('Finished')
        print(f'Runtime: {end-start}')
        print(f'Status: {problem.status}')
        print(f'Objective: {problem.value}')
        
        np.save(file=path+'completed_score_matrix.npy',arr = X.value) 
        
        return X.value
        
        
        
        