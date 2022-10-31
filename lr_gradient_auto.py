import torch
import torch.nn as nn

#design model
#construct loop, forward, backward, update weights


X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

#w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#def forward(x):
#    return w*x

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

#model = nn.Linear(input_size, output_size)
model = LinearRegression(input_size, output_size)


def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()


print(f'Prediction before traiing: f(5) = {model(X_test).item():.3f}')

learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    #prediction: forward pass
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradients
    l.backward()
    #dw = gradient(X, Y, y_pred)

    #update
    optimizer.step()

    optimizer.zero_grad()
    #w.grad.zero_()
    if epoch%10==0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss= {l:.8f}')

print(f'Prediction after traiing: f(5) = {model(X_test).item():.3f}')

#print(w)
