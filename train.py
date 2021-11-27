import torch
import torch.optim as optim
import dataset as ds
from  mirnet3 import Mirnet, CrossEntropyLossWithGaussianSmoothedLabels

net = Mirnet()
loss_fn = CrossEntropyLossWithGaussianSmoothedLabels()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_songs, test_songs = ds.get_train_test_songs()

for epoch in range(2):  # loop over the dataset multiple times
	for song in train_songs:
		# zero the parameter gradients
		running_loss=0.0
		optimizer.zero_grad()

		#load track data and convert to tensor
		x_data,y_data = datify_track(song)
		seq_in = torch.tensor(x,dtype=torch.float)
		labels = torch.tensor(y,dtype=torch.float)

		#foward and backward pass
		preds = net.forward(seq_in)
		loss = loss_fn.forward(preds, y_data)
		loss.backward()
		opimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')
