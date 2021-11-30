print("Test of standard out", flush=True)
import torch
import torch.optim as optim
import numpy as np
import dataset as ds
from save_load_model import *
from  mirnet3 import Mirnet, CrossEntropyLossWithGaussianSmoothedLabels2

def eval_test(net,test_songs): 
    print("Running on eval ", flush=True)
    song_accs = []
    for song in test_songs:
        x_data,y_data = ds.datify_track(song, one_hot=False)
        
        #split into chunks of seq_len frames and batch
        window_size = x_data.shape[1]
        num_classes = y_data.shape[1]
        num_batches = int(x_data.shape[0]/(seq_len*batch_size))
        x_data = x_data[:-(x_data.shape[0]%(seq_len*batch_size))]
        x_data = np.reshape(x_data,(num_batches,batch_size,1,seq_len,window_size))
        y_data = y_data[:-(y_data.shape[0]%(seq_len*batch_size))]
        y_data = np.reshape(y_data,(num_batches,batch_size,1,seq_len,1))
        
        #print(x_data.shape)
        #print(y_data.shape)

        num_correct = 0
        num_points = 0
        for batch_id in range(num_batches):
            seq_in = torch.tensor(x_data[batch_id],dtype=torch.float)
            labels = y_data[batch_id]

            #naive accuracy
            out = net.forward(seq_in)
            preds = net.predict(out)
            #preds = torch.argmax(preds,-1)
            preds = preds.cpu().detach().numpy()
            #preds = np.rehsape(preds,(-1,dataset.NUM_CLASSES))
            target = np.reshape(labels,(-1))
            num_correct += np.sum(len(np.where(preds == target)[0]))
            num_points += len(target)

        if(num_batches > 0): 
            print("Song Eval Accuracy: " + str(num_correct/num_points), flush=True)
            song_accs.append(num_correct/num_points)

    print("Average Validation Accuracy " + str(np.mean(song_accs)), flush=True)
    

print("Initializing network", flush=True)
net = Mirnet(num_class=ds.NUM_CLASSES)
loss_fn = CrossEntropyLossWithGaussianSmoothedLabels2(num_classes=ds.NUM_CLASSES)

optimizer = optim.Adam(net.parameters(), lr=0.002)

print("Reading Train List")
train_songs, test_songs = ds.get_train_test_songs()
        
seq_len = 31
batch_size = 64

print("GPU Status")
print(torch.cuda.is_available())
print(torch.cuda.device_count())

eval_test(net,test_songs)

for epoch in range(20):  # loop over the dataset multiple times
    i=0

    for song in train_songs:
        song_loss=0
        running_loss=0.0

        #load track data and convert to tensor
        print("Loading track " + song, flush=True)
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
        
        #print(x_data.shape)
        #print(y_data.shape)

        #import pdb; pdb.set_trace()

        for batch_id in range(num_batches):
            seq_in = torch.tensor(x_data[batch_id],dtype=torch.float)
            labels = torch.tensor(y_data[batch_id],dtype=torch.float)

            #foward and backward pass
            optimizer.zero_grad()
            preds = net.forward(seq_in)
            loss = loss_fn.forward(preds, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            #running_loss += loss.item()
            song_loss += loss.item()
            step_loss = loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, step_loss), flush=True)
            i+=1
        
        print('song loss: ' + str(song_loss/num_batches), flush=True)

    eval_test(net,test_songs)
    print("saving model", flush=True) 
    save(net, 'models/mirnet'+str(epoch)+'.pt')

print('Finished Training')
