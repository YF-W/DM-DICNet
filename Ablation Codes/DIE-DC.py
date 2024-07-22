import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as resnet_model


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, mid_chaennels, out_channels, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, mid_chaennels, kernel_size=7, padding=3, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            mid_chaennels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1,
                      0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.conv(x)


class UPConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UPConv, self).__init__()
        self.conv = nn.Sequential(

            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DilationConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilationConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1,
                      padding=dilation, dilation=dilation,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

def ring(image):
    batch_size, channel, height, width = image.shape

    padded_image = torch.zeros(image.shape, device=image.device)

    start_row = (height - height // 2) // 2
    end_row = start_row + height // 2
    start_col = (width - height // 2) // 2
    end_col = start_col + height // 2

    result_image = image.clone()
    result_image[:, :, start_row:end_row, start_col:end_col] = padded_image[:, :, start_row:end_row, start_col:end_col]

    return result_image


class ImageProcessor3(nn.Module):
    """
    cut half

    padding: False means not to padding
    """
    def __init__(self, padding=False):
        super().__init__()
        self.padding = padding


    def forward(self, image):
        _, _, height, width = image.shape
        padded_image = torch.zeros(image.shape, device=image.device)

        start_row = (height - height // 2) // 2
        end_row = start_row + height // 2
        start_col = (width - width // 2) // 2
        end_col = start_col + width // 2

        if self.padding:
            padded_image[:, :, start_row:end_row, start_col:end_col] = image[:,:, start_row:end_row, start_col:end_col]
        else:
            padded_image = image[:, :, start_row:end_row, start_col:end_col]

        return padded_image


class ImageProcessor3_2(nn.Module):
    """
    cut half

    padding: False means not to padding
    """
    def __init__(self, padding=False):
        super().__init__()
        self.padding = padding


    def forward(self, image):
        _, _, height, width = image.shape
        padded_image = torch.zeros(image.shape, device=image.device)

        start_row = 28
        end_row = 28 + 168
        start_col = 28
        end_col = 28 + 168

        if self.padding:
            padded_image[:, :, start_row:end_row, start_col:end_col] = image[:,:, start_row:end_row, start_col:end_col]
        else:
            padded_image = image[:, :, start_row:end_row, start_col:end_col]

        return padded_image    

def padding(image, need_shape):
    """
    Pad the input image to the desired shape.
    
    Args:
    image (torch.Tensor): The input image tensor of shape (batch_size, channels, height, width).
    need_shape (int): The desired shape for height and width (must be square).
    
    Returns:
    torch.Tensor: The padded image tensor of shape (batch_size, channels, need_shape, need_shape).
    """
    batch_size, c, h, w = image.shape

    pad_h = (need_shape - h) // 2
    pad_w = (need_shape - w) // 2

    padded_image = torch.zeros(batch_size, c, need_shape, need_shape, device=image.device)
    padded_image[:, :, pad_h:pad_h + h, pad_w:pad_w + w] = image

    return padded_image

def cutting(image, need_shape):
    """
    Crop the input image to the desired shape.
    
    Args:
    image (torch.Tensor): The input image tensor of shape (batch_size, channels, height, width).
    need_shape (int): The desired shape for height and width (must be square).
    
    Returns:
    torch.Tensor: The cropped image tensor of shape (batch_size, channels, target_height, target_width).
    """
    batch_size, c, h, w = image.shape

    assert need_shape <= h and need_shape <= w, "Target shape must be smaller than or equal to the input shape."

    start_h = (h - need_shape) // 2
    start_w = (w - need_shape) // 2
    end_h = start_h + need_shape
    end_w = start_w + need_shape

    cropped_image = image[:, :, start_h:end_h, start_w:end_w]

    return cropped_image

class DIE_DC(nn.Module):
    def __init__(self):
        super(DIE_DC, self).__init__()
        self.down = nn.MaxPool2d(2, 2)

        self.x_03_2 = ImageProcessor3_2()
        self.x_03 = ImageProcessor3()

        self.dilationConv_up_1 = DilationConv(3, 64, 3)
        self.dilationConv_up_2 = DilationConv(64, 128, 3)
        self.dilationConv_up_3 = DilationConv(128, 256, 3)
        self.dilationConv_up_4 = DilationConv(256, 512, 3)

        self.dilationConv_mid_1 = DilationConv(3, 64, 2)
        self.dilationConv_mid_2 = DilationConv(64, 128, 2)
        self.dilationConv_mid_3 = DilationConv(128, 256, 2)
        self.dilationConv_mid_4 = DilationConv(256, 512, 2)

        self.doubleConv_down_1 = DoubleConv(3, 64)
        self.doubleConv_down_2 = DoubleConv(64, 128)
        self.doubleConv_down_3 = DoubleConv(128, 256)
        self.doubleConv_down_4 = DoubleConv(256, 512)

        self.conv_1 = DoubleConv(512, 512)
        self.conv_2 = DoubleConv(1024, 512)
        self.conv_3 = DoubleConv(512, 512)

        self.dilationConv_up_5 = DilationConv(512, 256, 3)
        self.dilationConv_up_6 = DilationConv(256, 128, 3)
        self.dilationConv_up_7 = DilationConv(128, 64, 3)
        self.dilationConv_up_8 = DilationConv(64, 16, 3)

        self.dilationConv_mid_5 = DilationConv(512, 256, 2)
        self.dilationConv_mid_6 = DilationConv(256, 128, 2)
        self.dilationConv_mid_7 = DilationConv(128, 64, 2)
        self.dilationConv_mid_8 = DilationConv(64, 16, 2)

        self.doubleConv_down_5 = DoubleConv(512, 256)
        self.doubleConv_down_6 = DoubleConv(256, 128)
        self.doubleConv_down_7 = DoubleConv(128, 64)
        self.doubleConv_down_8 = DoubleConv(64, 16)

        self.up1 = UPConv(512, 512)
        self.up2 = UPConv(256, 256)
        self.up3 = UPConv(128, 128)
        self.up4 = UPConv(64, 64)
        self.conv1x1 = nn.Conv2d(709, 512, 1)

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(16, 48, (1, 224))
        self.finalConv = nn.Conv2d(48, 1, 1)

        self.conv197_512 = nn.Conv2d(197, 512, 1)
        self.conv197_256 = nn.Conv2d(197, 256, 1)
        self.conv197_128 = nn.Conv2d(197, 128, 1)
        self.conv197_64 = nn.Conv2d(197, 64, 1)
        self.conv197_3 = nn.Conv2d(197, 3, 1)


        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x):

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e1 = self.encoder1(e0)  # torch.Size([4, 64,112
        e2 = self.encoder2(e1)  # torch.Size([4, 128,56
        e3 = self.encoder3(e2)  # torch.Size([4, 256,28
        e4 = self.encoder4(e3)  # torch.Size([4, 512, 14

        x_up_1 = x
        x_down_1 = self.x_03(x)

        # first layer
        x_down_2 = self.doubleConv_down_1(x_down_1)
        x_up_2 = self.dilationConv_up_1(padding(x_down_1, x_up_1.shape[2])+x_up_1)

        x_up_2_p = self.down(x_up_2)
        x_down_2_p = self.down(x_down_2)

        # second layer
        x_down_3 = self.doubleConv_down_2(x_down_2_p+cutting(x_up_2_p, x_down_2_p.shape[2]))
        x_up_3 = self.dilationConv_up_2(padding(x_down_2_p, x_up_2_p.shape[2])+x_up_2_p)
        
        x_up_3_p = self.down(x_up_3)
        x_down_3_p = self.down(x_down_3)

        # third layer
        x_down_4 = self.doubleConv_down_3(x_down_3_p+cutting(x_up_3_p, x_down_3_p.shape[2]))
        x_up_4 = self.dilationConv_up_3(padding(x_down_3_p, x_up_3_p.shape[2])+x_up_3_p)

        x_up_4_p = self.down(x_up_4)
        x_down_4_p = self.down(x_down_4)

        # forth layer
        x_down_5 = self.doubleConv_down_4(x_down_4_p+cutting(x_up_4_p, x_down_4_p.shape[2]))
        x_up_5 = self.dilationConv_up_4(padding(x_down_4_p, x_up_4_p.shape[2])+x_up_4_p)
        
        x_up = self.down(x_up_5) # torch.Size([4, 512, 14, 14])
        x_down = self.down(x_down_5) # torch.Size([4, 512, 7, 7])
        
        x = x_up + padding(x_down, x_up.shape[2])
        x = self.conv_1(x)
        x = torch.cat((x, e4),dim=1)
        x = self.conv_2(x)


        # decoder
        # input
        x_down_6, x_mid_6, x_up_6 = x, x, x

        # layer 1
        x_up_6 = F.interpolate(x_up_6, scale_factor=2, mode='bilinear', align_corners=True)
        x_down_6 = cutting(F.interpolate(x_down_6, mode="bilinear", align_corners=True, scale_factor=2), x_down_5.shape[2])
        e4 = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=True)

        x_down_7 = self.doubleConv_down_5(ring(x_down_5) + x_down_6)
        x_up_7 = self.dilationConv_up_5(x_up_6+padding(x_mid_6, x_up_6.shape[2])+ring(x_up_5)+e4)

        x_down_7 = F.interpolate(x_down_7, scale_factor=2, mode='bilinear', align_corners=True)
        x_up_7 = F.interpolate(x_up_7, scale_factor=2, mode='bilinear', align_corners=True)
        e3 = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=True)

        x_down_8 = self.doubleConv_down_6(x_down_7 + ring(x_down_4) + cutting(x_up_7, x_down_7.shape[2]))
        x_up_8 = self.dilationConv_up_6(x_up_7+padding(x_down_7, x_up_7.shape[2])+ring(x_up_4)+e3)

        x_down_8 = F.interpolate(x_down_8, scale_factor=2, mode='bilinear', align_corners=True)
        x_up_8 = F.interpolate(x_up_8, scale_factor=2, mode='bilinear', align_corners=True)
        e2 = F.interpolate(e2, scale_factor=2, mode='bilinear', align_corners=True)

        x_down_9 = self.doubleConv_down_7(x_down_8 + ring(x_down_3) + cutting(x_up_8, x_down_8.shape[2]))
        x_up_9 = self.dilationConv_up_7(x_up_8+padding(x_down_8, x_up_8.shape[2])+ring(x_up_3)+e2)

        e1 = F.interpolate(e1, scale_factor=2, mode='bilinear', align_corners=True)
        x_down_9 = F.interpolate(x_down_9, scale_factor=2, mode='bilinear', align_corners=True)
        x_up_9 = F.interpolate(x_up_9, scale_factor=2, mode='bilinear', align_corners=True)

        x_down_10 = self.doubleConv_down_8(x_down_9 + ring(x_down_2) + cutting(x_up_9, x_down_9.shape[2]))
        x_up_10 = self.dilationConv_up_8(x_up_9+padding(x_down_9, x_up_9.shape[2])+ring(x_up_2)+e1)

        x_skip = torch.cat((padding(x_down_10, x_up_10.shape[2]), padding(x_down_10, x_up_10.shape[2]), x_up_10), dim=1)
        x_attn = self.sigmoid(self.conv(x_up_10))
        x = torch.mul(x_attn, x_skip)
        x = x + x_skip
        x = self.finalConv(x)

        return x

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224).cuda()
    model = DIE_DC().cuda()
    out = model(x)
    print(out.shape)
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print(f'Flops: {flops}, params: {params}')
