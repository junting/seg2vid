import torch
import torch.nn as nn


class my_vgg(nn.Module):
    def __init__(self, vgg):
        super(my_vgg, self).__init__()
        self.vgg = vgg
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, img):
        # x = img.unsqueeze(0)
        x1 = self.vgg.features[0](img)
        x2 = self.vgg.features[1](x1) # relu
        x3 = self.vgg.features[2](x2)
        x4 = self.vgg.features[3](x3) # relu
        x5 = self.avgpool(x4)

        x6 = self.vgg.features[5](x5)
        x7 = self.vgg.features[6](x6) # relu
        x8 = self.vgg.features[7](x7)
        x9 = self.vgg.features[8](x8) # relu
        x10 = self.avgpool(x9)

        x11 = self.vgg.features[10](x10)
        x12 = self.vgg.features[11](x11) # relu
        x13 = self.vgg.features[12](x12)
        x14 = self.vgg.features[13](x13) # relu
        x15 = self.vgg.features[14](x14)
        x16 = self.vgg.features[15](x15) # relu
        x17 = self.vgg.features[16](x16)
        x18 = self.vgg.features[17](x17) # relu
        x19 = self.avgpool(x18)

        x20 = self.vgg.features[19](x19)
        x21 = self.vgg.features[20](x20) # relu
        x22 = self.vgg.features[21](x21)
        x23 = self.vgg.features[22](x22) # relu
        x24 = self.vgg.features[23](x23)
        x25 = self.vgg.features[24](x24) # relu
        x26 = self.vgg.features[25](x25)
        x27 = self.vgg.features[26](x26) # relu

        return x2, x4, x7, x9, x12, x14, x16, x18, x21, x23, x25, x27
        # return x1, x3, x6, x8, x11, x13, x15, x17, x20, x22, x24, x26


if __name__ == '__main__':
    import torchvision
    vgg19 = torchvision.models.vgg19(pretrained=True)
    vgg = my_vgg(vgg19)
    tmp = torch.randn((32, 3, 16, 16))
    x2, x4, x7, x9, x12, x14, x16, x18, x21, x23, x25, x27 = vgg(tmp)
    print(x27.size())
