import BreakfastNaive as BF
import torch
import torch.nn as nn
from model import NeuralNet
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 2048
hidden_size = 1024
num_classes = 48
num_epochs = 5  # TODO
batch_size = 100  # TODO increase
learning_rate = 0.001  # TODO

# DATASET
visual_feat_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\bf_kinetics_feat"
text_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\groundTruth"
map_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\mapping.txt"

visual_feat_path_train = os.path.join(visual_feat_path, "train")
text_path_train = os.path.join(text_path, "train")

visual_feat_path_test = os.path.join(visual_feat_path, "test")
text_path_test = os.path.join(text_path, "test")

train_dataset = BF.BreakfastNaive(visual_feat_path_train, text_path_train, map_path)
test_dataset = BF.BreakfastNaive(visual_feat_path_test, text_path_test, map_path)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# MODEL
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TRAIN
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, data_batch in enumerate(train_loader):
        vis_feats = data_batch['vis_feats'].to(device)
        labels = data_batch['labels'].to(device)  # TODO dont send one of them to gpu and see what happens?

        # fwd pass
        outputs = model(vis_feats.float())
        loss = criterion(outputs, labels.long())

        # bkwd pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# TEST
with torch.no_grad():
    correct = 0
    total = 0
    for data_batch in test_loader:
        vis_feats = data_batch['vis_feats'].to(device)
        labels = data_batch['labels'].to(device)

        outputs = model(vis_feats.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #TODO print confusion matrix

    print('Accuracy over entire test set: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
