import torch
import torch.optim as optim
import numpy as np
import dataset as ds
#from save_load_model import *
from  mirnet3 import Mirnet, CrossEntropyLossWithGaussianSmoothedLabels2

net = Mirnet(num_class=ds.NUM_CLASSES)
loss_fn = CrossEntropyLossWithGaussianSmoothedLabels2(num_classes=ds.NUM_CLASSES)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_songs, test_songs = ds.get_train_test_songs()
        
seq_len = 31
batch_size = 64

for epoch in range(2):  # loop over the dataset multiple times
    i=0
    for song in train_songs:
        # zero the parameter gradients
        running_loss=0.0
        optimizer.zero_grad()

        #load track data and convert to tensor
        x_data,y_data = ds.datify_track(song)
        #import pdb;pdb.set_trace()

        #split into chunks of seq_len frames and batch
        window_size = x_data.shape[1]
        num_classes = y_data.shape[1]
        num_batches = int(x_data.shape[0]/(seq_len*batch_size))
        x_data = x_data[:-(x_data.shape[0]%(seq_len*batch_size))]
        x_data = np.reshape(x_data,(num_batches,batch_size,1,seq_len,window_size))
        y_data = y_data[:-(y_data.shape[0]%(seq_len*batch_size))]
        y_data = np.reshape(y_data,(num_batches,batch_size,1,seq_len,num_classes))
        
        print(x_data.shape)
        print(y_data.shape)

        #import pdb; pdb.set_trace()

        for batch_id in range(num_batches):
            seq_in = torch.tensor(x_data[batch_id],dtype=torch.float)
            labels = torch.tensor(y_data[batch_id],dtype=torch.float)

            #foward and backward pass
            preds = net.forward(seq_in)
            loss = loss_fn.forward(preds, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            i+=1
#save(net, 'trainedModel')
print('Finished Training')
