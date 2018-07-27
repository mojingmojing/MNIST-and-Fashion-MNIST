## https://www.kaggle.com/c/digit-recognizer

Last time I played with the MNIST dataset, in this post I will train a classifier to classifier images of digits into different digit classes. 


```python
# import some libraries
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

I split the training data into two parts, 90% of the training data becomes the new training set, and 10% becomes the dev set. Training data is used to build the model, and dev set is to determine if the model we just trained is a good model. The reason why we need dev set is because the model may easily overfit on the training set, but it cannot perform well outside the domain of the training set. 


```python
data = pd.read_csv("train.csv")
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


I wrote a function to evaluate the model. It breaks the dataset into batches, and make predictions based on batched data. We count and accumulate the number of correct predictions. 


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

The following neural network is copied from https://github.com/pytorch/examples/blob/master/mnist/main.py . 
It works surprisingly well on this particular image recognition task. I wonder if it also works this well on other tasks. I might write another blog post for other tasks in the future. 


```python
# define a neural network, convolutional neural network
class Net(nn.Module):
    '''
        network structure borrowed from https://github.com/pytorch/examples/blob/master/mnist/main.py
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(x.size(0), 1, int(image_height), int(image_width)) # B * 1 * 28 * 28
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # B * 10 * 12 * 12
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # B * 20 * 4 * 4
        x = x.view(-1, 320) 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# instantiate the model
model = Net()
```

We start training using Adam Optimizer. The optimizer helps us reduce the loss function we defined on the correct label and our predicted labels, the goal is to reduce the loss as much as possible. 


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
        torch.save(model.state_dict(), "model.th")
    else:
        learning_rate *= 0.8
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
```

    dev acc: 0.8745238095238095
    save the model
    dev acc: 0.9238095238095239
    save the model
    dev acc: 0.9116666666666666
    dev acc: 0.9397619047619048
    save the model
    dev acc: 0.9452380952380952
    save the model
    dev acc: 0.9435714285714286
    dev acc: 0.9514285714285714
    save the model
    dev acc: 0.9502380952380952
    dev acc: 0.9488095238095238
    dev acc: 0.9607142857142857
    save the model
    dev acc: 0.9592857142857143
    dev acc: 0.9647619047619047
    save the model
    dev acc: 0.959047619047619
    dev acc: 0.9666666666666667
    save the model
    dev acc: 0.9645238095238096
    dev acc: 0.9707142857142858
    save the model
    dev acc: 0.9685714285714285
    dev acc: 0.9726190476190476
    save the model
    dev acc: 0.9697619047619047
    dev acc: 0.9692857142857143


Make predictions! and write the prediction file to the submision file. This gives me over 96% of accuracy. 


```python
test_data = pd.read_csv("test.csv")
model.load_state_dict(torch.load("model.th"))
def predict(model, data):
    model.eval()
    correct_count = 0.
    total_count = 0.
    all_preds = []
    for i in range(0, data.shape[0], BATCH_SIZE):
        batch_data = data.iloc[i:i+BATCH_SIZE, :].values
        x = batch_data[:, :] # 32 * 784
        x = Variable(torch.from_numpy(x), volatile=True).float()
        pred = model(x) # B*10
        pred = np.argmax(pred.data.numpy(), 1)
        all_preds.append(pred[None, :])
    all_preds = np.concatenate(all_preds, 0)
    return all_preds
all_preds = predict(model, test_data)
all_preds = all_preds.reshape(-1)
all_preds.shape
```




    (28000,)




```python
with open("submission.csv", "w") as f:
    f.write("ImageId,Label\n")
    for i in range(all_preds.shape[0]):
        f.write("{},{}\n".format(i+1, all_preds[i]))
```
