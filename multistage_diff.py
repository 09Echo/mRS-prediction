import torch
import torch.nn as nn
from text_pre import BertModel_sentence,BertModel
from method import vit_encoder_b
import torch.nn.functional as F

class SAMVIT(nn.Module):
    def __init__(self,imgencoder_path=None,num_classes=2,reduce_dim=64,textencoder_path=None):
        super().__init__()
        img_dim = 768
        text_dim = 768
        self.reduce_dim = reduce_dim
        self.image_encoder = vit_encoder_b(num_classes,imgencoder_path)
        self.difftext_encoder = BertModel_sentence(textencoder_path)
        self.patient_encoder = BertModel(textencoder_path)
        self.fc = nn.Linear(self.reduce_dim * 4, num_classes)

        self.reduce_mlp = nn.Linear(text_dim,self.reduce_dim)
        self.conv = nn.Conv2d(in_channels=img_dim,out_channels=self.reduce_dim * 2, kernel_size=1)
        self.muti = MultiscaleText(reduce_dim=self.reduce_dim)
        self.avgpool  = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x,prompt,information):
        x_flip = torch.flip(x,dims=[3])
        diff_emb = self.difftext_encoder(prompt).unsqueeze(1) #(B,1,768)
        patient_emb = self.patient_encoder(information) #(B,num_sentence=12,768)
        patient_emb = self.reduce_mlp(patient_emb)

        img_emb = self.image_encoder(x, x_flip, diff_emb)

        img_emb = self.conv(img_emb)
        weight_l, weight_g = self.muti(img_emb,patient_emb)
        B, C, H, W = img_emb.shape
        feature = torch.cat((img_emb.reshape(B, H, W, C), weight_l, weight_g),dim=3)
        feature = self.avgpool(feature.reshape(B, -1, H, W))
        if feature.shape[0] == 1:
            feature = torch.squeeze(feature)
            feature = torch.unsqueeze(feature,0)
        else:
            feature = torch.squeeze(feature)
        feature = self.fc(feature)
        return feature

class MultiscaleText(nn.Module):
    def __init__(self, reduce_dim=64):
        super(MultiscaleText,self).__init__()
        self.reduce_dim = reduce_dim

        self.dw_conv1 = DepthwiseSeparableConv(self.reduce_dim * 2, self.reduce_dim, kernel_size=3, padding=1)
        self.dw_conv2 = DepthwiseSeparableConv(self.reduce_dim * 2, self.reduce_dim, kernel_size=5, padding=2)

    def forward(self,x,information):
        Local_prompt = information
        Global_prompt = torch.mean(information,dim=1,keepdim=True)
        x1 = self.dw_conv1(x)
        B, C, H, W = x1.shape
        x_l = x1.reshape(B, H * W, C)
        att_l = (x_l @ Local_prompt.transpose(-2, -1)).softmax(dim=1) #(B,196,20)
        weight_l = (torch.sum(att_l,axis=2).softmax(dim=-1)).unsqueeze(2) * x_l

        x2 = self.dw_conv2(x)
        x_g = x2.reshape(B, H * W, C)
        att_g = (x_g @ Global_prompt.transpose(-2, -1)).softmax(dim=1)  # (B,196,1)
        weight_g = att_g * x_g

        return weight_l.reshape(B,H,W,C),weight_g.reshape(B,H,W,C)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return F.relu(x)




