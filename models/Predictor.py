import torch.nn as nn
class Predictor(nn.Module):
    def __init__(self, img_feats, img_encoder, class_num, use_feats=(4,), fc_dims=(512,)):
        in_dim = self.img_encoder.out_dim
        fc_layers = []
        if len(fc_dims) > 0:
            for _, fc_dim in enumerate(fc_dims):
                fc_layer = nn.Sequential(nn.Linear(in_dim, fc_dim),nn.BatchNorm1d(fc_dim), nn.ReLU())
                fc_layers.append(fc_layer)
                in_dim = fc_dim
        fc_layers.append(nn.Linear(in_dim, class_num))
        self.fc_layers = nn.Sequential(*fc_layers)
        self.img_feats = img_feats
    def forward(self, x):
        pred_scores = self.fc_layers(self.img_feats)
        return pred_scores