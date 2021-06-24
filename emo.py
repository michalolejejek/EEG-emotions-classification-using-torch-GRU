# First I make data analisis and visualization in jupter notebook (you can see it in analitics file)
# When I see what we have I make this model and I trained it
#
#
#
#-------------------------------------------------------

# import modules
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sbn


def get_default_device():
    # return GPU is avalible or cpu when you dont have gpu
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    return data.to(device, non_blocking=True)

# bulding model 
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_size).float()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, h = self.gru(x)
        out = self.fc(self.relu(out))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

#----------------

#init device using get_default_device function
device = get_default_device()

# Read the data
data = pd.read_csv('emotions.csv')
labels = data['label'].value_counts()

# Preprocessing data


# do not change original dataset
data_copy = data.copy()

# label encoding
data['label'] = data['label'].astype('category').cat.codes

# split data into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], random_state=42, test_size=0.2, stratify=data['label'])

y_test = y_test.to_numpy()

# Bulding our model

inputs_array = X_train.to_numpy()
targets_array = y_train.to_numpy()

# convert to tensors
inputs = to_device(torch.FloatTensor(inputs_array),device)
targets = to_device(torch.FloatTensor(targets_array), device)

batch_size = 64 # defining batch size

input_size = len(X_train.columns)
output_size = len(y_train.unique())
hidden_dim = 128
n_layers = 2
# init hyperparameters
n_epochs = 430
# init model
model = Model(input_size, output_size, hidden_dim, n_layers)
to_device(model, device)
 

# Training model

#define loss and otimizer
losses = []
l_rates = [1e-1, 1e-2, 1e-3, 1e-4]
l_r_i = 2
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), l_rates[l_r_i])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.01)
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() 
    output, hidden = model(inputs.unsqueeze(0))
    loss = criterion(output.squeeze(0).float(), targets.long())
    loss_detached = loss.detach().cpu().clone().numpy()
    losses.append(loss_detached)

    loss.backward() 
    optimizer.step()
#     scheduler.step(loss)

    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))


# let's make a prediction

test_data = to_device(torch.FloatTensor(X_test.to_numpy()).unsqueeze(0), device)
output = model(test_data)[0]
output = output.squeeze(0)
output_ = output.detach().cpu().clone()
predictions = np.array(torch.argmax(output_, 1, keepdim=True))

# And at least we can see classification report

print(classification_report(y_test, predictions))


# MODEL VALIDATION



c_m = confusion_matrix(predictions, y_test)

plt.figure(figsize=(10, 10))
sbn.heatmap(c_m, annot=True, cmap='YlGnBu', fmt='g', yticklabels=list(labels.index), xticklabels=list(labels.index))

