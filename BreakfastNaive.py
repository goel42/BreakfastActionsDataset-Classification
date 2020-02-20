import numpy as np
import os
from itertools import groupby
from torch.utils.data import Dataset


def __get_map__(path, inv=False):
    """
    returns
    if inv == False : a dictionary of label_strings to integer
    else: a dictionary of integer to label_strings
    """

    m = {}
    with open(path, "r") as f:
        text = f.readlines()

    for line in text:
        temp = line.strip().split()
        num, activity = int(temp[0]), temp[1]
        if inv:
            m[num] = activity
        else:
            m[activity] = num

    return m


def __get_labels__(path):
    with open(path, "r") as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    return labels


def __get_data__(vis_feat_path, action_label_path, stoi):
    vis_feats_all = []
    labels_all = []

    vis_files = os.listdir(vis_feat_path)
    label_files = os.listdir(action_label_path)

    for vf, lf in zip(vis_files, label_files):
        feats = np.load(os.path.join(vis_feat_path, vf))
        labels = __get_labels__(os.path.join(action_label_path, lf))
        labels = [stoi[label] for label in labels]
        labels = np.array(labels)

        # thee are both cases where label < feats and feats < labels
        # Temp Debug. #TODO remove after dataset cleaning
        if feats.shape[0] != labels.shape[0]:
            # if labels.shape[0] > feats.shape[0] :
            #     print("FATAL! There are files where frames are more than visual features")
            #     print(feats.shape, labels.shape)
            #     print(vf)

            min_len = min(feats.shape[0], labels.shape[0])
            feats = feats[:min_len, :]
            labels = labels[:min_len]

        assert feats.shape[0] == labels.shape[0]

        # caveat: itertools.groupby works only when the input is sorted.
        # i.e. when all identical keys are together.
        # this works well in our case because we want ( SIL at start) and (SIL at end) to be grouped separately
        for label, group in groupby(zip(labels, feats), lambda x: x[0]):
            grouped_feats = [group_item[1] for group_item in group]
            grouped_feats = np.array(grouped_feats)

            feat_maxpooled = np.amax(grouped_feats, axis=0)
            vis_feats_all.append(feat_maxpooled)

            labels_all.append(label)
    return {"vis_feats": vis_feats_all, "labels": labels_all}


class BreakfastNaive(Dataset):

    def __init__(self, vis_feat_path, action_label_path, map_path):
        self.visual_feat_path = vis_feat_path
        self.action_label_path = action_label_path
        self.stoi_map = __get_map__(map_path)
        self.itos_map = __get_map__(map_path, inv=True)
        self.data = __get_data__(self.visual_feat_path,
                                 self.action_label_path, self.stoi_map)

    def __len__(self):
        assert len(self.data["vis_feats"]) == len(self.data["labels"])
        return len(self.data["vis_feats"])

    def __getitem__(self, idx):
        return {"vis_feats": self.data["vis_feats"][idx],
                "labels": self.data["labels"][idx]}
