import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model(n_input_features=6)

FILE = 'model.pth'
FILE2 = 'model2.pth'
#torch.save(model, FILE)
model = torch.load(FILE)
model.eval()

for param in model.parameters():
    print(param)


torch.save(model.state_dict(), FILE2)

loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE2))
loaded_model.eval()
for param in model.parameters():
    print(param)


for param in model.parameters():
    print(param)

learning_Rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_Rate)
print(model.state_dict())
print(optimizer.state_dict())

checkpoint = {
        "epoch":90,
        "model_state": model.state_dict(),
        "optim_State": optimizer.state_dict()
        }

#torch.save(checkpoint, "checkpoint.pth")

loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr = 0)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])

print(optimizer.state_dict())



