import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def shapealter(A):
	number_songs,number_features = A.shape
	A = A.reshape(number_songs,1,number_features)
	return A

def standardize(X):
	mu = np.mean(X, axis = 0)
	std = np.std(X, axis = 0)
	return mu, std

class testNet(nn.Module):
	def __init__(self):
		super(testNet, self).__init__()
		self.fc1 = nn.Linear(445, 100)
		#self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(100, 14)

	def forward(self, x):
		# If the size is a square you can only specify a single number
		x = F.sigmoid(self.fc1(x)) #hidden1
		#x = F.relu(self.fc2(x)) #hidden2
		x = F.log_softmax(self.fc3(x)) #output
		return x


	def Train(self, x, y,lr = 0.1 ,n_epochs = 150,batch_size = 10):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print(device)

		self.to(device)
		criterion = nn.CrossEntropyLoss()
		opt = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-3) #momentum #nestrov momentum

		x = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad = True)
		y = Variable(torch.from_numpy(y).type(torch.LongTensor), requires_grad = False)

		losses = []
		for e in range(n_epochs):
		#self.train()
			for beg in range(0,x.size(0),batch_size):
				x_batch = x[beg:beg+batch_size]
				y_batch = y[beg:beg+batch_size]

				opt.zero_grad() # What does it do??
				y_hat = self.forward(x_batch) # Forward prop
				loss = criterion(y_hat, y_batch)
				loss.backward()
				opt.step()

			los = loss.data.numpy()
			print(e, los)
			losses.append(los)

		y_pred = self.forward(x).detach().numpy()
		y = y.numpy()

		predicted = np.argmax(y_pred,1)
		c = (predicted == y).squeeze()
		#print(y_pred.shape, y.shape)
		accuracy = 100*np.sum(c)/len(y_train)
		#print(y_pred[:50],y[:50])
		print(accuracy)
		torch.save(self.state_dict(), 'deepModel.pth')
		return losses

if __name__ == "__main__":
	y = np.load("label_array.npy")
	X = np.load("data_array.npy")

	np.random.seed(5)
	index = np.random.permutation(X.shape[0])

	X = X[index]
	y = y[index]

	n_train = int(0.8*X.shape[0])
	X_train = X[:n_train]
	y_train = y[:n_train]

	X_test = X[n_train:]
	y_test = y[n_train:]

	mu, sigma = standardize(X_train[:,:14])

	X_train[:,:14] = (X_train[:,:14] - mu)/sigma
	X_test[:,:14] = (X_test[:,:14] - mu)/sigma

	model = testNet()
	print(X_train.shape, y_train.shape)
	losses = model.Train(X_train, y_train ,lr = 0.00009,n_epochs = 1000,batch_size = 30)

	X_test_torch = Variable(torch.from_numpy(X_test).type(torch.FloatTensor), requires_grad = True)

	y_hat_test = model.forward(X_test_torch)
	pred = np.argmax(y_hat_test.detach().numpy(), 1)
	c_test = (pred == y_test)
	accuracy_test = 100*np.sum(c_test)/len(y_test)
	print(accuracy_test)

	plt.plot(losses)
	plt.savefig('./Results(losses)/89.png')   # save the figure to file
	plt.show()
