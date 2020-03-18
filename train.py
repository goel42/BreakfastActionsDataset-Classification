# import BreakfastNaive as BF
import BreakfastNaiveFS as BF

#NaiveFS removes SIL, has w2v for text labels, uniq, has tempcheck to select activities and sources,

import torch
import torch.nn as nn
from model import NeuralNet
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 400
hidden_size = 128
num_classes = 48
num_epochs = 200  # TODO
batch_size = 100  # TODO increase
learning_rate = 0.00001  # TODO

# DATASET for website (GT from FS though)
# visual_feat_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\bf_kinetics_feat"
# text_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\groundTruth"
# map_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\mapping.txt"

# DATASET for fs dataset
visual_feat_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\Breakfast_fs\data_maxpool"
text_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\Breakfast_fs\groundTruth_maxpool_clean"
map_path = r"C:\Users\dcsang\PycharmProjects\embedding\breakfast\Breakfast_fs\splits\mapping_clean.txt"


visual_feat_path_train = os.path.join(visual_feat_path, "train")
text_path_train = os.path.join(text_path, "train")

visual_feat_path_test = os.path.join(visual_feat_path, "test")
text_path_test = os.path.join(text_path, "test")


print("Starting to load training data")
train_dataset = BF.BreakfastNaiveFS(visual_feat_path_train, text_path_train, map_path, rm_SIL=False)
# train_dataset = BF.BreakfastNaive(visual_feat_path_train, text_path_train, map_path)

print("Training set loaded")
test_dataset = BF.BreakfastNaiveFS(visual_feat_path_test, text_path_test, map_path, rm_SIL=False)
# test_dataset = BF.BreakfastNaive(visual_feat_path_test, text_path_test, map_path)
print("Test set loaded")

print("Dataset size")
print(len(train_dataset))
print(len(test_dataset))

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
    correct_predictions = 0
    total_predictions = 0
    for i, data_batch in enumerate(train_loader):
        vis_feats = data_batch['vis_feats'].to(device)
        labels = data_batch['labels'].to(device)  # TODO dont send one of them to gpu and see what happens?
        # print("DEBUG vis feats", vis_feats.size())
        # print("DEBUG labels", labels.size())


        # fwd pass
        outputs = model(vis_feats.float())
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        # print("DEBUG outputs! ", outputs.size())
        loss = criterion(outputs, labels.long())
        # print("DEBUG loss! ", loss, loss.size())


        # bkwd pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    print('Epoch [{}/{}], Accuracy: {:.4f}'.format(epoch + 1, num_epochs,100 * correct_predictions / total_predictions ))

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
