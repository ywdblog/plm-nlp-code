import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 网络上的一个例子 

# 定义自定义数据集
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

# 定义神经网络模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 准备数据
train_data = torch.tensor([[0.5], [1.0], [1.5], [2.0]])
train_labels = torch.tensor([0, 0, 1, 1])
test_data = torch.tensor([[0.8], [1.2], [1.6]])
test_labels = torch.tensor([0, 0, 1])

train_dataset = MyDataset(train_data, train_labels)
test_dataset = MyDataset(test_data, test_labels)

# 定义超参数
input_size = 1
hidden_size = 10
num_classes = 2
learning_rate = 0.01
batch_size = 2
num_epochs = 100

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型、损失函数和优化器
model = MLP(input_size, hidden_size, num_classes)
model.to(torch.device('cpu')) # construct model on CPU
criterion = nn.CrossEntropyLoss() #set loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # set optimizer

# 训练模型
for epoch in range(num_epochs): #iterate over epochs
    for batch_data, batch_labels in train_loader: #iterate through the dataloader
        # 前向传播
        outputs = model(batch_data) #move data to device
        loss = criterion(outputs, batch_labels) #forward pass

        # 反向传播和优化
        optimizer.zero_grad() #set gradients to zero
        loss.backward() #compute loss gradients
        optimizer.step() #compte gradients (backpropagation step) and update parameters

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在测试集上评估模型
model.eval() #set model to evaluation（评估） mode
with torch.no_grad(): #disable gradient calculation 
    # 为什么要禁用梯度计算？
    # 因为在测试集上只需要前向传播，不需要反向传播，所以禁用梯度计算可以节省内存，提高速度
    correct = 0
    total = 0
    for batch_data, batch_labels in test_loader: #iterate through the test data loader
        outputs = model(batch_data) #forward pass
        _, predicted = torch.max(outputs.data, 1) #get predicted labels
        total += batch_labels.size(0) #count number of samples
        correct += (predicted == batch_labels).sum().item() #count number of correct predictions

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.2%}') #print accuracy

torch.save(model.state_dict(), 'model.ckpt') #保存模型参数
cpkt = torch.load('model.ckpt') #加载模型参数
model.load_state_dict(cpkt) #加载模型参数

# 使用训练好的模型进行预测
new_data = torch.tensor([[1.1], [1.8], [2.5]])
with torch.no_grad():
    predictions = model(new_data)
    _, predicted_labels = torch.max(predictions.data, 1)

print("Predictions:")
for i in range(new_data.size(0)):
    print(f'Input: {new_data[i][0]:.1f}  Predicted Label: {predicted_labels[i]}')
