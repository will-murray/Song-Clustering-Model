Two scripts implemented:

	1. 
		initialize_music_space data for data preprocessing. 
		This gets converts the Million Song Subset from a collection of .h5 to a single .npy file vector space.
		.npy is numpy's file format and is apparently efficient.
		
		the feature vectors in v1_music_space.npy contains 105 dimensions (see initialize_music_space -> grab_song() )

	2. 
		music_space.py, the interface which will be used by the game engine. See the file for documentation

TODO:
	Test efficiency of neareast neighbours as dimensions increase
		fast with 105 dimensions


