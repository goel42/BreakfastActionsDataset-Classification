#[Insert some nice header]
..and some pictures


##Introduction

##Installation

##Dataset

##Preprocessing

##Model

##Experiments


##TODO
1. We operate on segment level (eg: pour_milk). The visual feature for a segment is obtained by max-pooling the i3D features of constituent frames. Use a Sequence model (eg: RNN/LSTM) to encode features for a segment (from constituent frames) so that temporal info can be encoded
2. Add graphs from training/testing iterations 