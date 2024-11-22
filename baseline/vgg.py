import torchvision
import torch
vgg16 = torchvision.models.vgg16(pretrained = False,num_classes=2)

if __name__ == '__main__':
    x = torch.rand(10,3,224,224)
    model = vgg16
    total = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('+ Number of Backbone Params: %.4f(e6)' % (total / 1e6))
    y = model(x)
    print(y)