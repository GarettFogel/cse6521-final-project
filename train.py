import sys
import torch
import torch.optim as optim
import numpy as np
import dataset as ds
from save_load_model import *
from  mirnet3 import Mirnet, CrossEntropyLossWithGaussianSmoothedLabels2
from evaluation_functions import *

USE_F_LAYER=False
if(len(sys.argv) < 2):
    arg = "noname"
else:
    arg = sys.argv[1]
print(arg)
print("Using Fourier Layer? " + str(USE_F_LAYER),flush=True)
if(USE_F_LAYER):
    preprocess=False
    use_f_layer=True
    name="yesf" 
else:
    preprocess=True
    use_f_layer=False
    name="nof"

def eval_test(net,test_songs): 
    print("Running on eval ", flush=True)
    song_accs = []
    all_preds = []
    all_targets = []
    for song in test_songs:
        x_data,y_data = ds.datify_track(song, one_hot=False, preprocess=preprocess)
        
        #split into chunks of seq_len frames and batch
        window_size = x_data.shape[1]
        num_classes = y_data.shape[1]
        num_batches = int(x_data.shape[0]/(seq_len*batch_size))
        if(num_batches==0): 
            print("Not enough data for song: " + str(x_data.shape[0]))
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
            preds = preds.cpu().detach().numpy()
            all_preds.append(preds)
            target = np.reshape(labels,(-1))
            all_targets.append(target)
            num_correct += np.sum(len(np.where(preds == target)[0]))
            num_points += len(target)

        if(num_batches > 0): 
            print("Song Eval Accuracy: " + str(num_correct/num_points), flush=True)
            song_accs.append(num_correct/num_points)
            

    print("Average Validation Accuracy " + str(np.mean(song_accs)), flush=True)
    all_preds = np.concatenate(all_preds).astype(int)
    all_targets = np.concatenate(all_targets).astype(int)
    #import pdb; pdb.set_trace()
    print("rca: " + str(rca(all_preds, all_targets)))
    print("oa: " + str(oa(all_preds, all_targets)))
    print("rpa: " + str(rpa(all_preds, all_targets)))
    print(VoiceConfusionMatrix(all_preds,all_targets))
    with np.printoptions(threshold=np.inf):
        conf_mat = np.round_(PitchConfusionMatrix(all_preds,all_targets),3)
        for row in range(conf_mat.shape[0]):
            print(",".join(str(val) for val in conf_mat[row]))
        #print(PitchConfusionMatrix(all_preds,all_targets))
    

print("Initializing network", flush=True)
net = Mirnet(num_class=ds.NUM_CLASSES, use_f_layer=use_f_layer)
loss_fn = CrossEntropyLossWithGaussianSmoothedLabels2(num_classes=ds.NUM_CLASSES)

optimizer = optim.Adam(net.parameters(), lr=0.0002)

print("Reading Train List")
train_songs, test_songs = ds.get_train_test_songs()
        
seq_len = 31
batch_size = 32

print("GPU Status")
print(torch.cuda.is_available())
print(torch.cuda.device_count())

eval_test(net,test_songs)

for epoch in range(10):  # loop over the dataset multiple times
    print("Epoch: " + str(epoch), flush=True)
    i=0

    for song in train_songs:
        song_loss=0
        running_loss=0.0

        #load track data and convert to tensor
        print("Loading track " + song, flush=True)
        x_data,y_data = ds.datify_track(song, preprocess=preprocess, one_hot=True, blur=True)
        #import pdb;pdb.set_trace()
        
        #split into chunks of seq_len frames and batch
        window_size = x_data.shape[1]
        num_classes = y_data.shape[1]
        num_batches = int(x_data.shape[0]/(seq_len*batch_size))
        x_data = x_data[:-(x_data.shape[0]%(seq_len*batch_size))]
        x_data = np.reshape(x_data,(num_batches,batch_size,1,seq_len,window_size))
        y_data = y_data[:-(y_data.shape[0]%(seq_len*batch_size))]
        y_data = np.reshape(y_data,(num_batches,batch_size,1,seq_len,num_classes))
        #y_data = np.reshape(y_data,(num_batches,batch_size,1,seq_len))
        
        #print(x_data.shape)
        #print(y_data.shape)

        #import pdb; pdb.set_trace()

        batch_ids = np.arange(num_batches)
        np.random.shuffle(batch_ids)
        for batch_id in batch_ids:
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
       
        if(num_batches>0): 
            print('song loss: ' + str(song_loss/num_batches), flush=True)

    eval_test(net,test_songs)
    print("saving model", flush=True) 
    save(net, 'models/mirnet'+name+arg+str(epoch)+'.pt')

print('Finished Training')
