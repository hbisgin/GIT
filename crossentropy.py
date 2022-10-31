import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss
def foo():
    return "NO"

#y must be one hot coded
#class 0 [1, 0, 0]
#class 1 [0, 1, 0]

Y = np.array([1, 0, 0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)

print(f'Loss 1 numpy: {l1:.4f}')
print(f'Loss 2numpy: {l2:.4f}')


#pytorch
#already applies LogSoftmax. No need for Softmaz here. 
#class labels are not one-hot
# y pred raw scores

loss = nn.CrossEntropyLoss()
Y = torch.tensor([2,0,1])
#number of samples * n classes 1x3
Y_pred_good  = torch.tensor([[0.1, 1.0, 2.1],[2.0, 1.0, 0.1], [.1, 3.0, 0.1] ])
Y_pred_bad  = torch.tensor([[2.5, 1.0, 0.1], [.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item(), l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print(predictions1, predictions2)
