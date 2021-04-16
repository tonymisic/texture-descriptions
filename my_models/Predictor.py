import numpy as np
import torch, torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models, transforms
from torchvision.models.resnet import Bottleneck, model_urls

class Predictor(nn.Module):
    def __init__(self, class_num, backbone='resnet101', pretrained_backbone=True, use_feats=(4,),
                 fc_dims=(512,)):
        super(Predictor, self).__init__()
        self.img_encoder = ResnetEncoder(backbone, pretrained_backbone, use_feats)

        in_dim = self.img_encoder.out_dim
        fc_layers = []
        if len(fc_dims) > 0:
            for _, fc_dim in enumerate(fc_dims):
                fc_layer = nn.Sequential(nn.Linear(in_dim, fc_dim),
                                         nn.BatchNorm1d(fc_dim),
                                         nn.ReLU())
                fc_layers.append(fc_layer)
                in_dim = fc_dim
        fc_layers.append(nn.Linear(in_dim, class_num))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        img_feats = self.img_encoder(x)
        pred_scores = self.fc_layers(img_feats)
        return pred_scores

class ResNet(models.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        self.feat_dims = np.array([64, 256, 512, 1024, 2048])

    def extract_feats(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = self.maxpool(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]

    def extract_flat_feats(self, x):
        feats = self.extract_feats(x)
        flat_feats = []
        for fi, f in enumerate(feats):
            if fi <= 4:
                ff = self.avgpool(f)
                ff = torch.flatten(ff, 1)
                flat_feats.append(ff)
        return flat_feats

def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

class ResnetEncoder(nn.Module):
    def __init__(self, backbone='resnet101', pretrained_backbone=True, use_feats=(4,)):
        super(ResnetEncoder, self).__init__()
        self.use_feats = use_feats
        if backbone == 'resnet101':
            self.backbone = resnet101(pretrained=pretrained_backbone)
        else:
            raise NotImplementedError
        feats_dim = 0
        for fi in use_feats:
            feats_dim += self.backbone.feat_dims[fi]
        self.out_dim = feats_dim
        return
    def forward(self, x):
        feats = self.backbone.extract_flat_feats(x)
        feat = torch.cat([feats[fi] for fi in self.use_feats], dim=1)
        return feat