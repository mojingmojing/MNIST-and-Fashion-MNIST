## https://www.kaggle.com/zalando-research/fashionmnist

In the last blog post, I applied the CNN model to fashion MNIST. Not surprisingly, the model does not achieve as high accuracy as it did on the MNIST handwritten digit recongnition task. 

In this post, I applied the method of [Cross Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)), hoping to get a better model for fashion MNIST task. 


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
```

## Standard Convolutional Neural Network 

borrowed from https://github.com/pytorch/examples/blob/master/mnist/main.py


The CNN model is exactly the same as before.


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


```python
# define a neural network, convolutional neural network
class Net(nn.Module):
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


```

## Cross Validation

Cross validation happens here. I apply the so-called k-fold cross validation. I evenly split the training set into 10 folds. Each time of training, I keep one fold as the dev set and the remaining 9 sets as the training set. Same as before, I do early stopping on the dev set while training. This process is repeated 10 times, so we get 10 models. 


```python
num_dev_data = int(num_data * 0.1)
for split in range(10):
    dev_index = data.index.isin(list(range(num_dev_data*split, num_dev_data*(split+1))))
    train_data = data[~dev_index]
    dev_data = data[dev_index]

    # instantiate the model
    model = Net()
    
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
            torch.save(model.state_dict(), "model-cross-validate/model-{}.th".format(split))
        else:
            learning_rate *= 0.8
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
```

    dev acc: 0.5945
    save the model
    dev acc: 0.6318333333333334
    save the model
    dev acc: 0.6103333333333333
    dev acc: 0.6651666666666667
    save the model
    dev acc: 0.6885
    save the model
    dev acc: 0.5778333333333333
    dev acc: 0.6818333333333333
    dev acc: 0.7003333333333334
    save the model
    dev acc: 0.6788333333333333
    dev acc: 0.6648333333333334
    dev acc: 0.7033333333333334
    save the model
    dev acc: 0.7135
    save the model
    dev acc: 0.7331666666666666
    save the model
    dev acc: 0.7343333333333333
    save the model
    dev acc: 0.725
    dev acc: 0.739
    save the model
    dev acc: 0.7388333333333333
    dev acc: 0.7133333333333334
    dev acc: 0.7321666666666666
    dev acc: 0.7421666666666666
    save the model
    dev acc: 0.7543333333333333
    save the model
    dev acc: 0.7735
    save the model
    dev acc: 0.8011666666666667
    save the model
    dev acc: 0.7731666666666667
    dev acc: 0.7756666666666666
    dev acc: 0.7836666666666666
    dev acc: 0.7848333333333334
    dev acc: 0.7823333333333333
    dev acc: 0.792
    dev acc: 0.8071666666666667
    save the model
    dev acc: 0.805
    dev acc: 0.811
    save the model
    dev acc: 0.8151666666666667
    save the model
    dev acc: 0.812
    dev acc: 0.8193333333333334
    save the model
    dev acc: 0.821
    save the model
    dev acc: 0.8203333333333334
    dev acc: 0.8236666666666667
    save the model
    dev acc: 0.8261666666666667
    save the model
    dev acc: 0.8235
    dev acc: 0.6121666666666666
    save the model
    dev acc: 0.6643333333333333
    save the model
    dev acc: 0.6398333333333334
    dev acc: 0.7278333333333333
    save the model
    dev acc: 0.7505
    save the model
    dev acc: 0.7623333333333333
    save the model
    dev acc: 0.7416666666666667
    dev acc: 0.774
    save the model
    dev acc: 0.7718333333333334
    dev acc: 0.7718333333333334
    dev acc: 0.7768333333333334
    save the model
    dev acc: 0.7818333333333334
    save the model
    dev acc: 0.773
    dev acc: 0.7905
    save the model
    dev acc: 0.7936666666666666
    save the model
    dev acc: 0.7868333333333334
    dev acc: 0.7965
    save the model
    dev acc: 0.8023333333333333
    save the model
    dev acc: 0.7978333333333333
    dev acc: 0.8038333333333333
    save the model
    dev acc: 0.7578333333333334
    save the model
    dev acc: 0.7113333333333334
    dev acc: 0.7541666666666667
    dev acc: 0.7586666666666667
    save the model
    dev acc: 0.777
    save the model
    dev acc: 0.7693333333333333
    dev acc: 0.7771666666666667
    save the model
    dev acc: 0.785
    save the model
    dev acc: 0.7751666666666667
    dev acc: 0.8023333333333333
    save the model
    dev acc: 0.7823333333333333
    dev acc: 0.7968333333333333
    dev acc: 0.8056666666666666
    save the model
    dev acc: 0.8006666666666666
    dev acc: 0.8043333333333333
    dev acc: 0.8033333333333333
    dev acc: 0.8125
    save the model
    dev acc: 0.8225
    save the model
    dev acc: 0.8171666666666667
    dev acc: 0.8195
    dev acc: 0.696
    save the model
    dev acc: 0.7623333333333333
    save the model
    dev acc: 0.7716666666666666
    save the model
    dev acc: 0.7575
    dev acc: 0.7511666666666666
    dev acc: 0.7775
    save the model
    dev acc: 0.7868333333333334
    save the model
    dev acc: 0.786
    dev acc: 0.7978333333333333
    save the model
    dev acc: 0.7988333333333333
    save the model
    dev acc: 0.7963333333333333
    dev acc: 0.7898333333333334
    dev acc: 0.8055
    save the model
    dev acc: 0.8066666666666666
    save the model
    dev acc: 0.8073333333333333
    save the model
    dev acc: 0.7965
    dev acc: 0.8061666666666667
    dev acc: 0.8111666666666667
    save the model
    dev acc: 0.7898333333333334
    dev acc: 0.8171666666666667
    save the model
    dev acc: 0.763
    save the model
    dev acc: 0.7731666666666667
    save the model
    dev acc: 0.749
    dev acc: 0.763
    dev acc: 0.7738333333333334
    save the model
    dev acc: 0.7695
    dev acc: 0.7938333333333333
    save the model
    dev acc: 0.7861666666666667
    dev acc: 0.8018333333333333
    save the model
    dev acc: 0.7965
    dev acc: 0.8006666666666666
    dev acc: 0.8088333333333333
    save the model
    dev acc: 0.8098333333333333
    save the model
    dev acc: 0.8091666666666667
    dev acc: 0.8128333333333333
    save the model
    dev acc: 0.8106666666666666
    dev acc: 0.8091666666666667
    dev acc: 0.8168333333333333
    save the model
    dev acc: 0.8176666666666667
    save the model
    dev acc: 0.8171666666666667
    dev acc: 0.6698333333333333
    save the model
    dev acc: 0.649
    dev acc: 0.6846666666666666
    save the model
    dev acc: 0.6343333333333333
    dev acc: 0.7278333333333333
    save the model
    dev acc: 0.7243333333333334
    dev acc: 0.6911666666666667
    dev acc: 0.7441666666666666
    save the model
    dev acc: 0.7496666666666667
    save the model
    dev acc: 0.7508333333333334
    save the model
    dev acc: 0.7548333333333334
    save the model
    dev acc: 0.7528333333333334
    dev acc: 0.734
    dev acc: 0.6988333333333333
    dev acc: 0.7516666666666667
    dev acc: 0.7485
    dev acc: 0.7665
    save the model
    dev acc: 0.7581666666666667
    dev acc: 0.7675
    save the model
    dev acc: 0.7748333333333334
    save the model
    dev acc: 0.609
    save the model
    dev acc: 0.6981666666666667
    save the model
    dev acc: 0.6923333333333334
    dev acc: 0.6818333333333333
    dev acc: 0.7121666666666666
    save the model
    dev acc: 0.7431666666666666
    save the model
    dev acc: 0.7245
    dev acc: 0.7381666666666666
    dev acc: 0.7655
    save the model
    dev acc: 0.761
    dev acc: 0.7695
    save the model
    dev acc: 0.7738333333333334
    save the model
    dev acc: 0.7681666666666667
    dev acc: 0.7798333333333334
    save the model
    dev acc: 0.7783333333333333
    dev acc: 0.7766666666666666
    dev acc: 0.7851666666666667
    save the model
    dev acc: 0.7801666666666667
    dev acc: 0.7841666666666667
    dev acc: 0.7883333333333333
    save the model
    dev acc: 0.5773333333333334
    save the model
    dev acc: 0.6366666666666667
    save the model
    dev acc: 0.5978333333333333
    dev acc: 0.6511666666666667
    save the model
    dev acc: 0.654
    save the model
    dev acc: 0.6378333333333334
    dev acc: 0.665
    save the model
    dev acc: 0.6828333333333333
    save the model
    dev acc: 0.7266666666666667
    save the model
    dev acc: 0.741
    save the model
    dev acc: 0.755
    save the model
    dev acc: 0.7453333333333333
    dev acc: 0.7588333333333334
    save the model
    dev acc: 0.7545
    dev acc: 0.7546666666666667
    dev acc: 0.7688333333333334
    save the model
    dev acc: 0.7705
    save the model
    dev acc: 0.7881666666666667
    save the model
    dev acc: 0.7796666666666666
    dev acc: 0.7856666666666666
    dev acc: 0.7033333333333334
    save the model
    dev acc: 0.639
    dev acc: 0.6898333333333333
    dev acc: 0.734
    save the model
    dev acc: 0.7081666666666667
    dev acc: 0.7305
    dev acc: 0.7301666666666666
    dev acc: 0.745
    save the model
    dev acc: 0.741
    dev acc: 0.757
    save the model
    dev acc: 0.7668333333333334
    save the model
    dev acc: 0.7661666666666667
    dev acc: 0.7663333333333333
    dev acc: 0.774
    save the model
    dev acc: 0.7768333333333334
    save the model
    dev acc: 0.7791666666666667
    save the model
    dev acc: 0.7828333333333334
    save the model
    dev acc: 0.7698333333333334
    dev acc: 0.7763333333333333
    dev acc: 0.783
    save the model


## load the models for testing purpose

During testing time, I load all ten models and keep them in a list. When making a prediction on a single instance, I apply each of the 10 models to give scores, and add the predicted scores up to make a single prediction. I do argmax over the summed scores to make the final prediction.  


```python
models = []
for split in range(10):
    model = Net()
    model.load_state_dict(torch.load("model-cross-validate/model-{}.th".format(split)))
    models.append(model)
```


```python
def test(models, data):
    '''
        args:
            data: 42000 * 785 matrix
    '''
    for model in models:
        model.eval()
    correct_count = 0.
    total_count = 0.
    for i in range(0, data.shape[0], BATCH_SIZE):
        
        batch_data = data.iloc[i:i+BATCH_SIZE, :].values
        x = batch_data[:, 1:] # 32 * 784
        y = batch_data[:, 0] # 32
        x = Variable(torch.from_numpy(x), volatile=True).float()
        y = Variable(torch.from_numpy(y), volatile=True)
        preds = 0.
        for model in models:
            preds += model(x)
        correct_count += torch.sum(torch.max(preds, 1)[1] == y).data[0]
        total_count += batch_data.shape[0]
    return correct_count, total_count
        
```


```python

test_data = pd.read_csv("fashion-mnist_test.csv")
correct_count, total_count = test(models, test_data)
print("test acc: {}".format(correct_count/total_count))
```

    test acc: 0.8267


There is another way to do it, we can apply each model to the prediction task, and do argmax to get a single prediction of each model. Then we do a vote among those ten models. The majority wins. 


```python
from scipy.stats import mode


def test_vote(models, data):
    '''
        args:
            data: 42000 * 785 matrix
    '''
    for model in models:
        model.eval()
    correct_count = 0.
    total_count = 0.
    for i in range(0, data.shape[0], BATCH_SIZE):
        
        batch_data = data.iloc[i:i+BATCH_SIZE, :].values
        x = batch_data[:, 1:] # 32 * 784
        y = batch_data[:, 0] # 32
        x = Variable(torch.from_numpy(x), volatile=True).float()
        y = Variable(torch.from_numpy(y), volatile=True)
        preds = []
        for model in models:
            preds.append(torch.max(model(x), 1)[1])
        preds = torch.cat(preds, 1).data.numpy()
        votes = mode(preds, axis=-1)[0].reshape(-1)
        y = y.data.numpy().reshape(-1)
        correct_count += (y == votes).sum()
        total_count += batch_data.shape[0]
    return correct_count, total_count
```


```python
test_data = pd.read_csv("fashion-mnist_test.csv")
correct_count, total_count = test_vote(models, test_data)
print("test acc with majority vote: {}".format(correct_count/total_count))
```

    test acc with majority vote: 0.8172


In this case, majority vote is worse than summing over the scores. 
