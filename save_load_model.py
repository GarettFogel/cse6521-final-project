import torch
import torch.optim as optim
import dataset as ds
from  mirnet3 import Mirnet

def save(model, PATH):
    torch.save(model, PATH)

def save_checkpoint(epoch, model, optimizer, loss, PATH):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)

def load(PATH):
    model = torch.load(PATH)
    return model

def load_checkpoint(PATH):
    model = Mirnet(num_class=ds.NUM_CLASSES)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
