import numpy as np
import os
from itertools import groupby
from torch.utils.data import Dataset
from word2vec.word2vec import load_word2vec


# debugging tool to only read given activities and sources
def tempcheck(filename):
    activities = BreakfastNaiveFS.activities
    sources = BreakfastNaiveFS.sources

    for src in sources:
        for act in activities:
            if act in filename and src in filename:
                return False

    # print(filename)
    return True


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


def __get_data__(vis_feat_path, action_label_path, stoi, rm_SIL):
    vis_feats_all = []
    labels_all = []

    vis_files = os.listdir(vis_feat_path)
    label_files = os.listdir(action_label_path)

    for vf, lf in zip(vis_files, label_files):
        if tempcheck(vf):
            continue
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
            if rm_SIL and label == 0:
                continue
            grouped_feats = [group_item[1] for group_item in group]
            grouped_feats = np.array(grouped_feats)

            feat_maxpooled = np.amax(grouped_feats, axis=0)
            vis_feats_all.append(feat_maxpooled)

            labels_all.append(label)
    # blah = (np.array(labels_all) != 0)
    # print("Debug: ", len(labels_all))
    # print("Debug: ", np.sum(blah))
    # print(blah)
    return {"vis_feats": vis_feats_all, "labels": labels_all}

class BreakfastNaiveFS(Dataset):
    sources = ["cam01", "cam02", "webcam01", "webcam02", "stereo01", "stereo02"]
    activities = ['cereals', 'coffee', 'friedegg', 'juice', 'milk', 'pancake', 'salat', 'sandwich', 'scrambledegg',
                  'tea']

    def __init__(self, vis_feat_path, action_label_path, map_path, rm_SIL=False):
        self.word2vecs = load_word2vec()
        self.visual_feat_path = vis_feat_path
        self.action_label_path = action_label_path
        self.stoi_map = __get_map__(map_path)
        self.itos_map = __get_map__(map_path, inv=True)
        self.data = __get_data__(self.visual_feat_path,
                                 self.action_label_path, self.stoi_map, rm_SIL)
        self.data['labels_w2v'] = self.__get_label_w2v__()
        self.uniq = self.__get_uniq_labels__(map_path, rm_SIL)

    def __len__(self):
        assert len(self.data["vis_feats"]) == len(self.data["labels"]) == len(self.data["labels_w2v"])
        return len(self.data["vis_feats"])

    def __getitem__(self, idx):
        return {"vis_feats": self.data["vis_feats"][idx],
                "labels": self.data["labels"][idx],
                "labels_w2v": self.data["labels_w2v"][idx]}

    def __get_label_w2v__(self):
        # CAVEATS: SILs would be pre-removed here unlike the other dataset class
        # CAVEATS: raw (below) contains numbers and not strings

        raw = self.data['labels']
        feat = []
        for caption_int in raw:
            caption = self.itos_map[caption_int]
            # caption = str(caption_int)
            max_pooled_w2v = self.__get_w2v__(caption)
            feat.append(max_pooled_w2v)
        return feat

    def __get_uniq_labels__(self, map_path, rm_SIL):
        labels = []
        w2v = []
        with open(map_path, "r") as f:
            text = f.readlines()

        for line in text:
            temp = line.strip().split()
            num, activity = int(temp[0]), temp[1]
            if rm_SIL and activity == "SIL":
                continue
            labels.append(num)
            w2v.append(self.__get_w2v__(activity))
            # w2v.append(self.__get_w2v__(str(num)))

        return {"labels": labels, "w2v": w2v}

    def __get_w2v__(self, caption):

        """if caption == SIL then sends back a vector of 0s"""

        # tokens_count = 2  # TODO
        tokens_count = len(caption.split("_"))
        vec_size = 200
        w2v = np.zeros([tokens_count, vec_size])

        if caption != 'SIL':
            for i, token in enumerate(caption.split("_")):
                w2v[i, :] = self.word2vecs[token]
        max_pooled_w2v = np.max(w2v, axis=0)
        return max_pooled_w2v
