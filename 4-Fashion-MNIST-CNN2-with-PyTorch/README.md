## https://www.kaggle.com/zalando-research/fashionmnist

I have tested my classic two layer convolutional neural network on fashion MNIST, and I also tried using cross validation with the model, which improves the accuracy quite a bit. 

In this post, I start modifying the model a little bit to make it work even better. 


```python
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
```

- Data preparation


```python
data = pd.read_csv("fashion-mnist_train.csv")
data.head()
image_size = data.iloc[:, 1:].values.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
image_width
num_data = data.shape[0]
num_train_data = int(num_data * 0.9)
train_data = data.iloc[:num_train_data]
dev_data = data.iloc[num_train_data:]
```

## Standard Convolutional Neural Network 

borrowed from https://github.com/pytorch/examples/blob/master/mnist/main.py



```python
BATCH_SIZE = 32
def eval(model, data):
    '''
        args:
            data: 42000 * 785 matrix
            
    '''
    model.eval()
    correct_count = 0.
    total_count = 0.
    for i in range(0, data.shape[0], BATCH_SIZE):
        batch_data = data.iloc[i:i+BATCH_SIZE, :].values
        x = batch_data[:, 1:] # 32 * 784
        y = batch_data[:, 0] # 32
        x = Variable(torch.from_numpy(x), volatile=True).float()
        y = Variable(torch.from_numpy(y), volatile=True)
        pred = model(x)
        loss = F.nll_loss(pred, y)
        correct_count += torch.sum(torch.max(pred, 1)[1] == y).data[0]
        total_count += batch_data.shape[0]
    return correct_count, total_count
        
```

This time, I add some extra convolutional filters with size 3 and 4. The old filters of size 5 are still kept there. To feed the results of both convolutional filters into the next level, I concatenated the inner results. 


```python
# define a neural network, convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv11 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv12 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv21 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv22 = nn.Conv2d(10, 20, kernel_size=4)
        self.conv12_drop = nn.Dropout2d()
        self.conv22_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(820, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x1 = x.view(x.size(0), 1, int(image_height), int(image_width)) # B * 1 * 28 * 28
        x1 = F.relu(F.max_pool2d(self.conv11(x1), 2)) # B * 10 * 12 * 12
        x1 = F.relu(F.max_pool2d(self.conv12_drop(self.conv12(x1)), 2)) # B * 20 * 4 * 4
        x2 = x.view(x.size(0), 1, int(image_height), int(image_width)) # B * 1 * 28 * 28
        x2 = F.relu(F.max_pool2d(self.conv21(x2), 2)) # B * 10 * 13 * 13
        x2 = F.relu(F.max_pool2d(self.conv22_drop(self.conv22(x2)), 2)) # B * 20 * 5 * 5
        
        x1 = x1.view(-1, 320) 
        x2 = x2.view(-1, 500)
        x = torch.cat([x1, x2], 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# instantiate the model
model = Net()
```


```python
learning_rate = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_acc = 0.
# start training
for epoch in range(20):
    model.train()
    for i in range(0, train_data.shape[0], BATCH_SIZE):
        batch_data = train_data.iloc[i:i+BATCH_SIZE, :].values
        x = batch_data[:, 1:] # 32 * 784
        y = batch_data[:, 0] # 32
        x = Variable(torch.from_numpy(x)).float()
        y = Variable(torch.from_numpy(y))
        pred = model(x)
        optimizer.zero_grad()
        loss = F.nll_loss(pred, y)
        loss.backward()
        optimizer.step()
    correct_count, total_count = eval(model, dev_data)
    acc = correct_count / total_count
    print("dev acc: {}".format(acc))
    if acc > best_acc:
        best_acc = acc
        print("save the model")
        torch.save(model.state_dict(), "model-experiment.th")
    else:
        learning_rate *= 0.8
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
```

    dev acc: 0.7813333333333333
    save the model
    dev acc: 0.7783333333333333
    dev acc: 0.794
    save the model
    dev acc: 0.7928333333333333
    dev acc: 0.8011666666666667
    save the model
    dev acc: 0.8033333333333333
    save the model
    dev acc: 0.8066666666666666
    save the model
    dev acc: 0.8051666666666667
    dev acc: 0.8143333333333334
    save the model
    dev acc: 0.7893333333333333
    dev acc: 0.8051666666666667
    dev acc: 0.8188333333333333
    save the model
    dev acc: 0.8233333333333334
    save the model
    dev acc: 0.819
    dev acc: 0.8148333333333333
    dev acc: 0.8251666666666667
    save the model
    dev acc: 0.82
    dev acc: 0.8263333333333334
    save the model
    dev acc: 0.8266666666666667
    save the model
    dev acc: 0.8271666666666667
    save the model



```python
model.load_state_dict(torch.load("model-experiment.th"))
test_data = pd.read_csv("fashion-mnist_test.csv")
correct_count, total_count = eval(model, test_data)
print("test acc: {}".format(correct_count/total_count))
```

    test acc: 0.8296


It is nice to see that the test results goes up compared to the previous network, which is not too surprising as we expand the network. 
