import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

anchor_generator = AnchorGenerator(sizes = ((32, 64, 128, 256, 512),),
                            aspect_ratios = ((0.5, 1.0, 2.0),))

# 사전 훈련된 모델의 feature map 데이터를 입력값으로 주어지도록 한다.
backbone = torchvision.models.resnet101(pretrained = True)

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names = ['0'],
                                                output_size = 7,
                                                sampling_ratio = 2)

mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names = ['0'],
                                                    output_size = 14,
                                                    sampling_ratio = 2)                            

model = MaskRCNN(backbone = backbone,
                num_classes = 2,
                box_roi_pool = roi_pooler,
                mask_roi_pool = mask_roi_pooler,
                rpn_anchor_generator = anchor_generator)

Model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)
Model.eval()

