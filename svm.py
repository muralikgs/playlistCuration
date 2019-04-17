from sklearn import svm
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    
    
    np.random.seed(680)
    dataDict = {}
    with open('spotifyDataset_onehot_encoded.pkl', 'rb') as f:
        dataDict = pickle.load(f)
        
    norm_feats = ['acousticness', 'danceability', 'duration', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'popularity', 'speechiness', 'tempo', 'time_signature', 'valence']
    means = np.zeros(len(norm_feats))
    for i,elem in enumerate(norm_feats):
            dataDict[elem] = (dataDict[elem] - np.mean(dataDict[elem]))/np.std(dataDict[elem])
            
    useless_feats = ['albumName','artistDetails','songID','songName','label']
    all_feats = []
    for keys in dataDict:
        all_feats.append(str(keys))        
    useful_feats = list(set(all_feats).difference(set(useless_feats)))
    useful_feats.remove('genres')    
    
    X = np.zeros((len(dataDict),446))
    
    for i,elem in enumerate(useful_feats):
        X[:,i] = dataDict[elem]

    for i in range(len(dataDict)):
        X[i,15:] = dataDict['genres'][i]

    Y = dataDict['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle = True)
    clf = svm.SVC(kernel = 'rbf', gamma='scale')
    clf.fit(X_train, Y_train) 
#    Y_pred = clf.predict(X_test)
#    accuracy1 = accuracy_score(Y_test,Y_pred)
    Y_pred_train1 = clf.predict(X_train)
    trainacc1 = accuracy_score(Y_train,Y_pred_train1)
    
    j=0
    count = 0
    noofsongs = {}
    for i in dataDict['label']:
        if i == j:
            count += 1
        elif i == j+1:
            noofsongs[int(j)] = count
            j = i
            count = 0
            count += 1
        
    totsongs = 0
    for i in range(len(noofsongs)):
        totsongs += noofsongs[i]
        
    noofsongs[int(max(dataDict['label']))] = len(dataDict)-totsongs

    lowerlim = min(noofsongs.values())
    newdata = np.zeros(((lowerlim*int(max(dataDict['label'])+1)),X.shape[1]+1)) 
    low=0
    XY = np.hstack((np.array(X),np.array(Y).reshape(1103,1)))
    for i in range(len(noofsongs)):
        pl = XY[low:low+noofsongs[i]]
        np.random.shuffle(pl)
        newdata[i*lowerlim:(i+1)*lowerlim] = pl[0:lowerlim]
        low += noofsongs[i]
    
    data = newdata[:,:-1]
    labels = newdata[:,-1]
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, shuffle = True)
    clf = svm.SVC(kernel = 'rbf', gamma='scale')
    clf.fit(X_train, Y_train) 
#    Y_pred = clf.predict(X_test)
#    accuracy2 = accuracy_score(Y_test,Y_pred)
    Y_pred_train2 = clf.predict(X_train)
    trainacc2 = accuracy_score(Y_train,Y_pred_train2)
  