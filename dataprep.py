import pickle
import numpy as np


if __name__ == "__main__":


    dataDict = {}
    with open('spotifyDataPrepared.pkl', 'rb') as f:
        dataDict = pickle.load(f)
       
    allgenres = []
    for i,elem in enumerate(dataDict['genres']):
        allgenres += elem
        
    allgenres = list(set(allgenres))
    allgenres.sort()
    
    genredict = {}
    for i in range(len(allgenres)):
        genredict[allgenres[i]] = 0
    
    onehot = np.zeros((len(dataDict),len(allgenres)))
    for i,elem in enumerate(dataDict['genres']):
        if elem == []:
            dataDict['genres'][i] = onehot[i]
            print(i)
            i += 1
            continue
        else:
            for genre in elem:
                genredict[genre] = 1
            onehot[i] = np.array(list(genredict.values()))
            genredict= dict.fromkeys(genredict, 0)
        dataDict['genres'][i] = onehot[i]
        
    filename = 'spotifyDataset_onehot_encoded.pkl' 
    with open(filename, 'wb') as pkl_file:
        pickle.dump(dataDict, pkl_file)