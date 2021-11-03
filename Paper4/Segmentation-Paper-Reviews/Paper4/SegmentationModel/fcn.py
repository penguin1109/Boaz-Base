from torch import nn
from Paper4.SegmentationModel._utils import _SimpleSegmentationModel
__all__ = ["FCN"]

class FCN(_SimpleSegmentationModel):
    """
        Implements a Fully-Convolutional Network for semantic segmentation.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction. 
# backbone model로부터 얻을 수 있는 prediction값
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding = 1, bias = False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]
# super()로 기반 클래스 초기화가 가능하다.
# 즉, super() 뒤에 .을 붙여서 method를 호출하는 방법이다.
        super(FCNHead, self).__init__(*layers)

model = FCNHead()
print(model)