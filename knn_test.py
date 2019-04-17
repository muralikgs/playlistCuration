import numpy as np 
from sklearn import neighbors

def standardize(X):
	mu = np.mean(X, axis = 0)
	sigma = np.std(X, axis = 0)

	return mu, sigma

if __name__ == '__main__':
	
	data = np.load('data_array.npy')
	labels = np.load('label_array.npy')

	np.random.seed(10)
	index = np.random.permutation(len(data))
	data = data[index]
	labels = labels[index]

	n_train = int(0.8*len(data))

	X_train = data[:n_train]
	y_train = labels[:n_train]

	X_test = data[n_train:]
	y_test = labels[n_train:]

	mu, sigma = standardize(X_train[:,:14])

	X_train[:,:14] = (X_train[:,:14] - mu)/sigma
	X_test[:,:14] = (X_test[:,:14] - mu)/sigma

	tr_acc = list()
	ts_acc = list()

	for k in range(1,101):
		neigh = neighbors.KNeighborsClassifier(n_neighbors=k)

		neigh.fit(X_train, y_train)
		
		y_pred = neigh.predict(X_train)
		acc_train = 100*np.sum(y_pred == y_train)/len(y_train)

		y_pred_test = neigh.predict(X_test)
		acc_test = 100*np.sum(y_pred_test == y_test)/len(y_test)

		print("k = ", k, " Train Acc: ", acc_train, " Test Acc: ", acc_test)

		tr_acc.append(acc_train)
		ts_acc.append(acc_test)
