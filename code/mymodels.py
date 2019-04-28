import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		self.n_hidden = 128
		self.n_hidden2 = 1024
		self.n_output = 5
		self.n_input = 3000
		self.hidden1 = nn.Linear(self.n_input, self.n_hidden)
		self.hidden2 = nn.Linear(self.n_hidden, self.n_hidden2)
		self.hidden3 = nn.Linear(self.n_hidden2, self.n_hidden)
		self.out = nn.Linear(self.n_hidden, self.n_output)

	def forward(self, x):
		x = torch.sigmoid(self.hidden1(x))
		x = torch.sigmoid(self.hidden2(x))
		x = torch.sigmoid(self.hidden3(x))
		x = self.out(x)
		return x

class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(16, 32, 5)
		self.fc1 = nn.Linear(in_features=32 * 747, out_features=128)
		#self.fc2 = nn.Linear(128, 128)
		self.dropout=nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(128, 5)
        
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 32 * 747)
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		#x = F.relu(self.fc2(x))
		#x = self.dropout(x)
		x = self.fc2(x)
		return x


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=2, batch_first=True, dropout=0.3)
		self.fc = nn.Linear(in_features=32, out_features=5)
        
	def forward(self, x):
		x, _ = self.rnn(x)
		x = self.fc(x[:, -1, :])
		return x
