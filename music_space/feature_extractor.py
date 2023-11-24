import music_space.hdf5_getters as h5
import numpy as np


def extractor_1(path):    
    """
    simple general features
    usage: IDK
    """

    song_file = h5.open_h5_file_read(path)

        
    X = [
        np.char.decode(h5.get_artist_name(song_file)),
        np.char.decode(h5.get_artist_id(song_file)),
        float(h5.get_artist_familiarity(song_file)),
        float(h5.get_artist_hotttnesss(song_file)),
        float(h5.get_danceability(song_file)),
        
        ]


    sid = h5.get_song_id(song_file)
    song_file.close()
    
    return [sid, X]

def extractor_2(path,TERMS_SET):
    """
    return a vector where the ith entry is
    the weight of the ith artist term for that
    song
    """
    TERMS_SET = TERMS_SET[1:41]
    print(TERMS_SET)
    return
    song_file = h5.open_h5_file_read(path)

    terms = [term for term in np.char.decode(h5.get_artist_terms(song_file))]
    weights = [weight for weight in h5.get_artist_terms_weight(song_file)]
    term_weight_dict = dict(zip(terms,weights))
    X = []
    for TERM in TERMS_SET:
        if TERM in term_weight_dict:
            X.append(term_weight_dict[TERM])
        else:
            X.append(0)


    sid = h5.get_song_id(song_file)
    song_file.close()
    return [sid, X]

