from collections import OrderedDict

from torch import nn
from torch.nn import functional as F

class _SimpleSegmentationModel(nn.module):
    __constatnts__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier = None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x) # Feature Map의 추출

        result = OrderedDict()
        