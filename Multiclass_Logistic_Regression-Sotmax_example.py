

import torch
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import pickle


"""
To train a linear classifier for multi-class classification, use softmax (instead of sigmoid, obviously!)

Reference-
https://diegoinacio.github.io/machine-learning-notebooks-page/pages/MCLR_PyTorch.html
"""


def data_loader_mnist(path_to_files):
        # Load training and test MNIST sets-
        train_set = datasets.MNIST(
                root = path_to_files, train = True,
                transform = None, download = True
                )

        test_set = datasets.MNIST(
                root = path_to_files, train = False,
                transform = None, download = True
                )

        X_train, y_train = train_set.data.numpy(), train_set.targets.numpy()
        X_test, y_test = test_set.data.numpy(), test_set.targets.numpy()

        # X_train.shape, y_train.shape
        # ((60000, 28, 28), (60000,))

        # X_test.shape, y_test.shape
        # ((10000, 28, 28), (10000,))

        # one hot encode target for multi-class classification-
        # MNIST = 10 classes [0-9]
        y_train_ohe = np.zeros((y_train.size, 10))
        y_train_ohe[np.arange(y_train.size), y_train] = 1
        y_test_ohe = np.zeros((y_test.size, 10))
        y_test_ohe[np.arange(y_test.size), y_test] = 1

        # y_train_ohe.shape, y_test_ohe.shape
        # ((60000, 10), (10000, 10))

        # Normalize by division with 255-
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # Conver to float32-
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train_ohe = y_train_ohe.astype(np.float32)
        y_test_ohe = y_test_ohe.astype(np.float32)

        # Flatten train samples into 1D vectors-
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

        # Convert to torch tensors-
        X_train = torch.Tensor(X_train)
        X_test = torch.Tensor(X_test)
        y_train_ohe = torch.Tensor(y_train_ohe)
        y_test_ohe = torch.Tensor(y_test_ohe)

        return X_train, X_test, y_train_ohe, y_test_ohe


X_train, X_test, y_train_ohe, y_test_ohe = data_loader_mnist(path_to_files = '/home/amajumdar/Downloads/.data/')

print(f"X_train.shape: {X_train.shape}, y_train_ohe.shape: {y_train_ohe.shape}")
print(f"X_test.shape: {X_test.shape}, y_test_ohe.shape: {y_test_ohe.shape}")
# X_train.shape: torch.Size([60000, 784]), y_train_ohe.shape: torch.Size([60000, 10])
# X_test.shape: torch.Size([10000, 784]), y_test_ohe.shape: torch.Size([10000, 10])


"""
# Visualize some sample MNIST images-
fig, axs = plt.subplots(3, 6, sharex = True, sharey = True)

np.random.seed(1234)
for ax in axs.ravel():
    rindex = np.random.randint(y_train.size)
    ax.imshow(X_train[rindex])
    # title label + one-hot
    title = '{} :: '.format(y_train[rindex]) 
    title += ''.join([str(int(e)) for e in y_train_ohe[rindex]]) 
    ax.set_title(title)
plt.grid(False)
plt.suptitle("Saple MNIST train images")
plt.show()
"""


# Specify training hyper-parameters-
num_epochs = 500
batch_size = 128

# Multi-class Logistic Regression (Softmax)-
num_examples, dims = X_train.shape

# Create a single dense layer model-
layer = torch.nn.Linear(in_features = dims, out_features = 10, bias = True)

# Initialize (Kaiming normal) weights and (zero) biases-
torch.nn.init.kaiming_normal_(layer.weight)
torch.nn.init.zeros_(layer.bias)

# Define softmax function-
softmx_fn = torch.nn.Softmax(dim = 1)

def crossentropy_cost_fn(preds, true_labels):
        # compute cross-entropy multi-class cost-
        return -torch.mean(torch.sum(torch.log(preds) * true_labels, dim = 1))

# Define gradient descent optimizer-
optimizer = torch.optim.Adam(params = layer.parameters(), lr = 0.01)


# Python3 dictionary for capturing training metrics-
train_history = {}


for epoch in range(1, num_epochs + 1):
        # Randomly select a batch of train images-
        rand_batch = np.random.choice(len(y_train_ohe), size = batch_size, replace = False)
        X = X_train[rand_batch, :]
        y = y_train_ohe[rand_batch]

        # Train image predictions-
        preds = softmx_fn(layer(X))

        optimizer.zero_grad()
        loss = crossentropy_cost_fn(preds = preds, true_labels = y)
        acc = torch.mean(1 - torch.abs(y - preds)) * 100

        # Compute test dataset metrics-

        # Randomly select a batch of test images-
        rand_batch_t = np.random.choice(len(y_test_ohe), size = batch_size, replace = False)
        X_t = X_test[rand_batch_t, :]
        y_t = y_test_ohe[rand_batch_t]

        # Test image predictions-
        preds_t = softmx_fn(layer(X_t))
        loss_t = crossentropy_cost_fn(preds = preds_t, true_labels = y_t)
        acc_t = torch.mean(1 - torch.abs(y_t - preds_t)) * 100

        # Compute partial-derivatives and perform 1 step of gradient descent-
        loss.backward()
        optimizer.step()

        # Store metrics-
        train_history[epoch] = {
                'loss': loss.item(), 'acc': acc.item(),
                'val_loss': loss_t.item(), 'val_acc': acc_t.item()
                }

        print(f"epoch: {epoch}, loss = {loss.item():.4f}, acc = {acc.item():.2f}%, "
              f"test loss = {loss_t.item():.4f} & test acc = {acc_t.item():.2f}%")

del X, y, X_t, y_t, acc, loss, acc_t, loss_t, loss, rand_batch, rand_batch_t


# Extract trained paraeters-
wts = layer.weight.detach().numpy()
biases = layer.bias.detach().numpy()

wts.shape, biases.shape
# ((10, 784), (10,))

# Visualize trained weights-
fig, axs = plt.subplots(1, 10, sharey = True)
for i in range(10):
    axs[i].imshow(wts[i].reshape(28, 28))
    axs[i].set_title(r'$W_{}$'.format(i))
plt.suptitle("Trained parameters")
plt.show()

# Visualize training and testing metrics-

plt.figure(figsize = (9, 8))
plt.plot(list(train_history[e]['loss'] for e in train_history.keys()), label = 'loss')
plt.plot(list(train_history[e]['val_loss'] for e in train_history.keys()), label = 'test loss')
plt.legend(loc = 'best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Linear classifier training loss")
plt.show()

plt.figure(figsize = (9, 8))
plt.plot(list(train_history[e]['acc'] for e in train_history.keys()), label = 'acc')
plt.plot(list(train_history[e]['val_acc'] for e in train_history.keys()), label = 'test acc')
plt.legend(loc = 'best')
plt.xlabel("epochs")
plt.ylabel("acc %")
plt.title("Linear classifier training accuracy")
plt.show()

