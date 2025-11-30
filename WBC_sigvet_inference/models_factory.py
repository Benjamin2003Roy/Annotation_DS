import time
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from eca_resnet import eca_resnet18, eca_resnet50




class EnsembleModel(nn.Module):   
    def __init__(self, modelA, modelB,NUM_CLASSES):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        
        self.classifier = nn.Linear(NUM_CLASSES * 2, NUM_CLASSES)
        
    def forward(self, x):
        x1,features = self.modelA(x)
        x2,features = self.modelB(x)

        x = torch.cat((x1, x2), dim=1)
        out = self.classifier(x)
        return out
    

class Resnet18(nn.Module):
    def __init__(self, num_classes=11):
        super(Resnet18, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.resnet = models.resnet18(pretrained=True)
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.resnet(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features
    
class Resnet18_ECA(nn.Module):
    def __init__(self, num_classes=11):
        super(Resnet18_ECA, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.resnet_eca = eca_resnet18(k_size=[3, 5, 7, 7], num_classes=1000, pretrained=True)
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.resnet_eca(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features

# class Resnet18_ECA_1(nn.Module):
#     def __init__(self, num_classes=11):
#         super(Resnet18_ECA, self).__init__()
#         self.norm = nn.BatchNorm2d(3)
#         self.resnet_eca = eca_resnet18(k_size=[3, 5, 7, 7], num_classes=1000, pretrained=True)
        
#         self.fc_layer = nn.Sequential(
#             nn.Linear(1000, 712),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(712, 512),  # Added intermediate layer
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = self.norm(x)
#         features = self.resnet_eca(x)
#         x = features.reshape(-1, 1000)
#         intermediate_features = self.fc_layer[:4](x)  # Extract intermediate features after 256 layer
#         x = self.fc_layer[4:](intermediate_features)  # Pass through the remaining layers
#         return x, intermediate_features

class Resnet50_ECA(nn.Module):
    def __init__(self, num_classes=11):
        super(Resnet50_ECA, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.resnet_eca = eca_resnet50(k_size=[3, 5, 5, 7], num_classes=1000, pretrained=True)
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.resnet_eca(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features
    

class EfficientNet_B0(nn.Module):
    def __init__(self, num_classes=9):
        super(EfficientNet_B0, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.efficientnet_b0(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features
    

class EfficientNet_B1(nn.Module):
    def __init__(self, num_classes=11):
        super(EfficientNet_B1, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.efficientnet_b1 = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.efficientnet_b1(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features
    
    
class EfficientNet_B3(nn.Module):
    def __init__(self, num_classes=11):
        super(EfficientNet_B3, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.efficientnet_b1 = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.efficientnet_b1(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features
    

class EfficientNet_V2(nn.Module):
    def __init__(self, num_classes=11):
        super(EfficientNet_V2, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.efficientnet_v2_s = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.efficientnet_v2_s(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features
    

class MOBILENET_V3_S(nn.Module):
    def __init__(self, num_classes=11):
        super(MOBILENET_V3_S, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.mobilenet_v3_s = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.mobilenet_v3_s(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features
    

class MOBILENET_V3_L(nn.Module):
    def __init__(self, num_classes=11):
        super(MOBILENET_V3_L, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.mobilenet_v3_l = models.mobilenet_v3_large()
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.mobilenet_v3_l(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features


class WBC_Ensemble_MOBILENET_V3_L(nn.Module):
    def __init__(self, num_classes=11):
        super(WBC_Ensemble_MOBILENET_V3_L, self).__init__()
        self.mobilenet_v3_l_1 = MOBILENET_V3_L()
        self.mobilenet_v3_l_2 = MOBILENET_V3_L()
        self.mobilenet_v3_l_3 = MOBILENET_V3_L()
        self.combination_layer = nn.Sequential(nn.Linear(3000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )
    def forward(self, x):
        out_1, feature_1 = self.mobilenet_v3_l_1(x)
        out_2, feature_2 = self.mobilenet_v3_l_2(x)
        out_3, feature_3 = self.mobilenet_v3_l_3(x)
        features = torch.cat([feature_1, feature_2, feature_3], dim = -1)
        x = features.reshape(-1, 3000)
        x = self.combination_layer(x)
        return x, [out_1, out_2, out_3]
    

class WBC_Ensemble_EfficientNet_B0_V1(nn.Module):
    def __init__(self, num_classes=11):
        super(WBC_Ensemble_EfficientNet_B0_V1, self).__init__()
        self.efficientnet_b0_1 = EfficientNet_B0()
        self.efficientnet_b0_2 = EfficientNet_B0()
        self.combination_layer = nn.Sequential(nn.Linear(2000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )
    def forward(self, x):
        out_1, feature_1 = self.efficientnet_b0_1(x)
        out_2, feature_2 = self.efficientnet_b0_2(x)
        features = torch.cat([feature_1, feature_2], dim = -1)
        x = features.reshape(-1, 2000)
        x = self.combination_layer(x)
        return x, [out_1, out_2]
    

class WBC_Ensemble_EfficientNet_B0_V2(nn.Module):
    def __init__(self, num_classes=11):
        super(WBC_Ensemble_EfficientNet_B0_V2, self).__init__()
        self.efficientnet_b0_1 = EfficientNet_B0()
        self.efficientnet_b0_2 = EfficientNet_B0()
        self.efficientnet_b0_3 = EfficientNet_B0()
        self.combination_layer = nn.Sequential(nn.Linear(3000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )
    def forward(self, x):
        out_1, feature_1 = self.efficientnet_b0_1(x)
        out_2, feature_2 = self.efficientnet_b0_2(x)
        out_3, feature_3 = self.efficientnet_b0_3(x)
        features = torch.cat([feature_1, feature_2, feature_3], dim = -1)
        x = features.reshape(-1, 3000)
        x = self.combination_layer(x)
        return x, [out_1, out_2, out_3]

    
class WBC_Ensemble_MOBILENET_V3_S(nn.Module):
    def __init__(self, num_classes=11):
        super(WBC_Ensemble_MOBILENET_V3_S, self).__init__()
        self.mobilenet_v3_s_1 = MOBILENET_V3_S()
        self.mobilenet_v3_s_2 = MOBILENET_V3_S()
        self.mobilenet_v3_s_3 = MOBILENET_V3_S()
        self.mobilenet_v3_s_4 = MOBILENET_V3_S()
        self.mobilenet_v3_s_5 = MOBILENET_V3_S()
        self.combination_layer = nn.Sequential(nn.Linear(5000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )
    def forward(self, x):
        out_1, feature_1 = self.mobilenet_v3_s_1(x)
        out_2, feature_2 = self.mobilenet_v3_s_2(x)
        out_3, feature_3 = self.mobilenet_v3_s_3(x)
        out_4, feature_4 = self.mobilenet_v3_s_4(x)
        out_5, feature_5 = self.mobilenet_v3_s_5(x)
        features = torch.cat([feature_1, feature_2, feature_3, feature_4, feature_5], dim = -1)
        x = features.reshape(-1, 5000)
        x = self.combination_layer(x)
        return x, [out_1, out_2, out_3, out_4, out_5]
    

class WBC_Ensemble_Resnet18(nn.Module):
    def __init__(self, num_classes=11):
        super(WBC_Ensemble_Resnet18, self).__init__()
        self.resnet_1 = Resnet18()
        self.resnet_2 = Resnet18()
        self.resnet_3 = Resnet18()
        self.combination_layer = nn.Sequential(nn.Linear(3000, 1000),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )

    def forward(self, x):
        out_1, feature_1 = self.resnet_1(x)
        out_2, feature_2 = self.resnet_2(x)
        out_3, feature_3 = self.resnet_3(x)
        features = torch.cat([feature_1, feature_2, feature_3], dim = -1)
        x = features.reshape(-1, 3000)
        x = self.combination_layer(x)
        return x, [out_1, out_2, out_3]
    


class VIT_B_16(nn.Module):
    def __init__(self, num_classes=11):
        super(VIT_B_16, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.resnet = models.vit_b_16(pretrained=True)
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.resnet(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features
    
class SWIN_V2_S(nn.Module):
    def __init__(self, num_classes=11):
        super(SWIN_V2_S, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.resnet = models.swin_v2_s(pretrained=True)
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      nn.Softmax(dim=1)
                                      )

    def forward(self, x):
        x = self.norm(x)
        features = self.resnet(x)
        x = features.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x, features