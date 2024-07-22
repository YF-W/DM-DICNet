import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as resnet_model


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation1=2, dilation2=3):
        super(DilatedConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=dilation1, dilation=dilation1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dilated_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=dilation1, dilation=dilation1, bias=False)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dilated_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=dilation2, dilation=dilation2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Dilated Convolution and 1x1 Convolution
        dilated_out = self.dilated_conv1(x)
        conv1x1_out = self.conv1x1(x)

        # Combine results
        combined = dilated_out + conv1x1_out
        combined = self.bn2(combined)

        # Second Dilated Convolution
        out = self.dilated_conv2(combined)
        out = self.relu(out)

        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2):
        super(ResidualBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                        bias=False) if in_channels != out_channels else None

    def forward(self, x):
        identity = self.conv1x1(x)
        identity = self.bn1(identity)

        out = self.conv1(identity)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.match_channels:
            x = self.match_channels(x)

        out += x
        out = self.relu(out)

        return out
    

class TripleKernelFourierConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, sigma=10, keep_fraction=0.1):
        super(TripleKernelFourierConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sigma = sigma
        self.keep_fraction = keep_fraction

        self.high_freq_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.high_freq_bn = nn.BatchNorm2d(out_channels)
        self.high_freq_relu = nn.ReLU(inplace=True)

        self.low_freq_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation * (kernel_size // 2),
                                       dilation=dilation, bias=False)
        self.low_freq_bn = nn.BatchNorm2d(out_channels)
        self.low_freq_relu = nn.ReLU(inplace=True)

        self.global_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation * (kernel_size // 2),
                                     dilation=dilation, bias=False)
        self.global_bn = nn.BatchNorm2d(out_channels)
        self.global_relu = nn.ReLU(inplace=True)

    def pad_to_power_of_two(self, x):
        B, C, H, W = x.shape
        H_pad, W_pad = 2 ** (H - 1).bit_length(), 2 ** (W - 1).bit_length()
        pad_h, pad_w = H_pad - H, W_pad - W
        padded_x = F.pad(x, (0, pad_w, 0, pad_h))
        return padded_x, (H, W)

    def fft_transform(self, images):
        images, original_shape = self.pad_to_power_of_two(images)
        f_transform = torch.fft.fft2(images, dim=(-2, -1))
        f_transform_shifted = torch.fft.fftshift(f_transform, dim=(-2, -1))
        return f_transform_shifted, original_shape

    def ifft_transform(self, f_transform_shifted, original_shape):
        f_ishift = torch.fft.ifftshift(f_transform_shifted, dim=(-2, -1))
        images_back = torch.fft.ifft2(f_ishift, dim=(-2, -1))
        images_back = images_back.real[..., :original_shape[0], :original_shape[1]]  # 裁剪回原始形状
        return images_back

    def high_low_frequency_split(self, f_transform_shifted, sigma):
        B, C, H, W = f_transform_shifted.shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        d = torch.sqrt(x * x + y * y)
        gaussian_filter = torch.exp(-d ** 2 / (2 * sigma ** 2)).to(f_transform_shifted.device)

        low_freq = f_transform_shifted * gaussian_filter
        high_freq = f_transform_shifted * (1 - gaussian_filter)

        return high_freq, low_freq

    def forward(self, x):
        f_transform_shifted, original_shape = self.fft_transform(x)

        high_freq, low_freq = self.high_low_frequency_split(f_transform_shifted, self.sigma)

        high_freq_image = self.ifft_transform(high_freq, original_shape)
        low_freq_image = self.ifft_transform(low_freq, original_shape)

        high_freq_out = self.high_freq_relu(self.high_freq_bn(self.high_freq_conv(high_freq_image)))
        low_freq_out = self.low_freq_relu(self.low_freq_bn(self.low_freq_conv(low_freq_image)))

        global_out = self.global_relu(self.global_bn(self.global_conv(x)))

        combined_low_global = low_freq_out + global_out

        out = high_freq_out + combined_low_global
        return out
    

class SpatialAttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAttentionFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # Concatenate the input features along the channel dimension
        x = torch.cat([x1, x2], dim=1)
        # Apply 3x3 convolution
        x = self.conv(x)

        # Apply sigmoid to get attention weights
        attention_weights = self.sigmoid(x)

        # Apply element-wise multiplication
        out = x1 * attention_weights

        return out
    

def split_input(input_tensor):
    # input_tensor shape: (batch_size, channels, height, width)
    batch_size, channels, height, width = input_tensor.shape

    # Zero tensor for padding
    zero_tensor = torch.zeros_like(input_tensor)

    # Define center and border masks with smoother transitions
    center_mask = zero_tensor.clone()
    border_mask = zero_tensor.clone()

    transition_width = height // 8
    center_mask[:, :, transition_width:height - transition_width, transition_width:width - transition_width] = 1
    border_mask[:, :, :transition_width, :] = 1
    border_mask[:, :, height - transition_width:, :] = 1
    border_mask[:, :, transition_width:height - transition_width, :transition_width] = 1
    border_mask[:, :, transition_width:height - transition_width, width - transition_width:] = 1

    # Extract center and border features
    center_feature = input_tensor * center_mask
    border_feature = input_tensor * border_mask

    return center_feature, border_feature, input_tensor


class ZeroPadding(nn.Module):
    def forward(self, x, target_shape):
        pad_h = (target_shape[2] - x.shape[2]) // 2
        pad_w = (target_shape[3] - x.shape[3]) // 2
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h))

class BilinearInterpolation(nn.Module):
    def forward(self, x, target_shape):
        return F.interpolate(x, size=(target_shape[2], target_shape[3]), mode='bilinear', align_corners=True)
class SymmetricPadding(nn.Module):
    def forward(self, x, target_shape):
        pad_h = (target_shape[2] - x.shape[2]) // 2
        pad_w = (target_shape[3] - x.shape[3]) // 2
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

class UpperModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpperModule, self).__init__()
        self.zero_padding = ZeroPadding()
        self.bilinear_interpolation = BilinearInterpolation()
        self.spatial_attention_fusion = SpatialAttentionFusion(in_channels * 2, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.residual_block = ResidualBlock(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.dilated_conv_block = DilatedConvBlock(out_channels, out_channels)
        self.triple_fft_conv = TripleKernelFourierConvolution(in_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels * 2)
        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.symmetric_padding = SymmetricPadding()

    def forward(self, border_feature, global_feature):
        padded_border_feature = self.symmetric_padding(border_feature, global_feature.shape)
        fused_feature = self.spatial_attention_fusion(padded_border_feature, global_feature)

        bn1_out = self.bn1(fused_feature)

        dilated_out = self.dilated_conv_block(bn1_out)
        triple_fft_out = self.triple_fft_conv(fused_feature)

        residual_out = self.residual_block(fused_feature)
        residual_bn_out = self.bn2(residual_out)

        combined_1 = torch.cat([bn1_out, triple_fft_out], dim=1)
        combined_2 = torch.cat([residual_bn_out, dilated_out], dim=1)
        combined_2_bn = self.bn3(combined_2)
        final_combined = torch.cat([combined_1, combined_2_bn], dim=1)
        final_out = self.final_conv(final_combined)
        final_out = self.final_bn(final_out)

        return final_out

class DownModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownModule, self).__init__()
        self.zero_padding = ZeroPadding()
        self.bilinear_interpolation = BilinearInterpolation()
        self.spatial_attention_fusion = SpatialAttentionFusion(in_channels * 2, in_channels)
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.residual_block1 = ResidualBlock(out_channels, out_channels)
        self.residual_block2 = ResidualBlock(out_channels, out_channels)
        self.residual_block3 = ResidualBlock(out_channels, out_channels)
        self.dilated_conv_block = DilatedConvBlock(out_channels, out_channels)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.symmetric_padding = SymmetricPadding()
        self.final_conv = nn.Conv2d(out_channels*4, out_channels, kernel_size=1, bias=False)

    def forward(self, center_feature, global_feature):
        # zero_padded_center_feature = self.bilinear_interpolation(center_feature, global_feature.shape)
        zero_padded_center_feature = self.symmetric_padding(center_feature, global_feature.shape)
        fusion_output = self.spatial_attention_fusion(zero_padded_center_feature, global_feature)
        fusion_output = self.conv1x1_1(fusion_output)

        residual1_out = self.residual_block1(fusion_output)
        residual2_out = self.residual_block2(residual1_out)

        residual3_out = self.residual_block3(fusion_output)

        conv1x1_out = self.conv1x1_2(fusion_output)

        dilated_out = self.dilated_conv_block(fusion_output)

        # combined1 = residual2_out + residual3_out + conv1x1_out + dilated_out
        combined1 = torch.cat([ residual2_out, residual3_out], dim=1)
        combined2 = torch.cat([conv1x1_out, dilated_out], dim=1)
        combined3 = torch.cat([combined1, combined2], dim=1)
        final_out = self.final_conv(combined3)

        final_out = self.final_bn(final_out)

        return final_out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SDRE(nn.Module):
    def __init__(self, in_channels):
        super(SDRE, self).__init__()
        self.upper_module = UpperModule(in_channels, in_channels)
        self.down_module = DownModule(in_channels, in_channels)
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, input_tensor):
        _, channel, _, _ = input_tensor.shape
        center_feature, border_feature, global_feature = split_input(input_tensor)
        output_upper_tensor = self.upper_module(border_feature, global_feature)
        output_down_tensor = self.down_module(center_feature, global_feature)

        # Attention mechanism
        attention_weights = self.channel_attention(output_upper_tensor + output_down_tensor)
        final_tensor = attention_weights * output_upper_tensor + (1 - attention_weights) * output_down_tensor

        return final_tensor

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
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


def ring(image):
    """
    Pads the input image with zeros to create a ring-shaped image.

    Parameters:
        image (torch.Tensor): The input image tensor of shape (batch_size, channel, height, width).

    Returns:
        torch.Tensor: The padded image tensor of shape (batch_size, channel, height, width).
    """
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
    Initializes the class with the specified padding parameter.

    Parameters:
        padding (bool): Indicates whether padding should be applied or not.

    Returns:
        None
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
            padded_image[:, :, start_row:end_row, start_col:end_col] = image[:, :, start_row:end_row, start_col:end_col]
        else:
            padded_image = image[:, :, start_row:end_row, start_col:end_col]

        return padded_image


class ImageProcessor3_2(nn.Module):
    """
    Initializes the class.

    Args:
        padding (bool, optional): If True, the class will pad the input. Defaults to False.
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
            padded_image[:, :, start_row:end_row, start_col:end_col] = image[:, :, start_row:end_row, start_col:end_col]
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


class SCHFE(nn.Module):
    def __init__(self, in_channels, time):
        super(SCHFE, self).__init__()
        self.time = time
        self.circul = nn.ModuleList()
        self.circul1 = nn.ModuleList()
        self.circul2 = nn.ModuleList()
        for index in range(time):
            self.circul.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels // (2 ** index), in_channels // (2 ** (index + 1)), kernel_size=3, stride=1,
                              padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(in_channels // (2 ** (index + 1)))
                )
            )
            self.circul1.append(
                nn.Sequential(
                    nn.Conv2d(in_channels // (2 ** index), in_channels // (2 ** (index + 1)), kernel_size=3, stride=1,
                              padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(in_channels // (2 ** (index + 1)))
                )
            )
            self.circul2.append(
                nn.Sequential(
                    nn.Conv2d(in_channels // (2 ** (index + 1)), in_channels // (2 ** (index + 1)), kernel_size=5, stride=1,
                              padding=2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(in_channels // (2 ** (index + 1)))
                )
            )
        self.endconv = nn.Conv2d(2 * in_channels - in_channels // (2 ** time), in_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        image = x
        for index in range(self.time):
            image1 = self.circul[index](image)
            image1 = F.interpolate(image1, scale_factor=2, mode='bilinear', align_corners=True)
            image2 = self.circul1[index](image)
            image = self.circul2[index](image1 + image2)
            x = torch.cat((x, image), dim=1)
        x = self.endconv(x)
        return x


class DM_DICNet(nn.Module):
    def __init__(self):
        super(DM_DICNet, self).__init__()
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

        self.circuldeep = SCHFE(512, 7)

        self.dilationConv_up_5 = DilationConv(512, 256, 3)
        self.LGE_5 = SDRE(512)
        self.dilationConv_up_6 = DilationConv(256, 128, 3)
        self.LGE_6 = SDRE(256)
        self.dilationConv_up_7 = DilationConv(128, 64, 3)
        self.LGE_7 = SDRE(128)
        self.dilationConv_up_8 = DilationConv(64, 16, 3)
        self.LGE_8 = SDRE(64)

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
        x_up_2 = self.dilationConv_up_1(padding(x_down_1, x_up_1.shape[2]) + x_up_1)

        x_up_2_p = self.down(x_up_2)
        x_down_2_p = self.down(x_down_2)

        # second layer
        x_down_3 = self.doubleConv_down_2(x_down_2_p + cutting(x_up_2_p, x_down_2_p.shape[2]))
        x_up_3 = self.dilationConv_up_2(padding(x_down_2_p, x_up_2_p.shape[2]) + x_up_2_p)

        x_up_3_p = self.down(x_up_3)
        x_down_3_p = self.down(x_down_3)

        # third layer
        x_down_4 = self.doubleConv_down_3(x_down_3_p + cutting(x_up_3_p, x_down_3_p.shape[2]))
        x_up_4 = self.dilationConv_up_3(padding(x_down_3_p, x_up_3_p.shape[2]) + x_up_3_p)

        x_up_4_p = self.down(x_up_4)
        x_down_4_p = self.down(x_down_4)

        # forth layer
        x_down_5 = self.doubleConv_down_4(x_down_4_p + cutting(x_up_4_p, x_down_4_p.shape[2]))
        x_up_5 = self.dilationConv_up_4(padding(x_down_4_p, x_up_4_p.shape[2]) + x_up_4_p)

        x_up = self.down(x_up_5)  # torch.Size([4, 512, 14, 14])
        x_down = self.down(x_down_5)  # torch.Size([4, 512, 7, 7])

        x = x_up + padding(x_down, x_up.shape[2])
        x = self.conv_1(x)
        x = torch.cat((x, e4), dim=1)
        x = self.conv_2(x)
        x = self.circuldeep(x)

        # decoder
        # input
        x_down_6, x_mid_6, x_up_6 = x, x, x

        # layer 1
        x_up_6 = F.interpolate(x_up_6, scale_factor=2, mode='bilinear', align_corners=True)
        x_down_6 = cutting(F.interpolate(x_down_6, mode="bilinear", align_corners=True, scale_factor=2),
                           x_down_5.shape[2])
        e4 = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=True)

        x_down_7 = self.doubleConv_down_5(ring(x_down_5) + x_down_6)
        x_up_7 = self.dilationConv_up_5(x_up_6 + padding(x_mid_6, x_up_6.shape[2]) + ring(x_up_5) + self.LGE_5(e4))

        x_down_7 = F.interpolate(x_down_7, scale_factor=2, mode='bilinear', align_corners=True)
        x_up_7 = F.interpolate(x_up_7, scale_factor=2, mode='bilinear', align_corners=True)
        e3 = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=True)

        x_down_8 = self.doubleConv_down_6(x_down_7 + ring(x_down_4) + cutting(x_up_7, x_down_7.shape[2]))
        x_up_8 = self.dilationConv_up_6(x_up_7 + padding(x_down_7, x_up_7.shape[2]) + ring(x_up_4) + self.LGE_6(e3))

        x_down_8 = F.interpolate(x_down_8, scale_factor=2, mode='bilinear', align_corners=True)
        x_up_8 = F.interpolate(x_up_8, scale_factor=2, mode='bilinear', align_corners=True)
        e2 = F.interpolate(e2, scale_factor=2, mode='bilinear', align_corners=True)

        x_down_9 = self.doubleConv_down_7(x_down_8 + ring(x_down_3) + cutting(x_up_8, x_down_8.shape[2]))
        x_up_9 = self.dilationConv_up_7(x_up_8 + padding(x_down_8, x_up_8.shape[2]) + ring(x_up_3) + self.LGE_7(e2))

        e1 = F.interpolate(e1, scale_factor=2, mode='bilinear', align_corners=True)
        x_down_9 = F.interpolate(x_down_9, scale_factor=2, mode='bilinear', align_corners=True)
        x_up_9 = F.interpolate(x_up_9, scale_factor=2, mode='bilinear', align_corners=True)

        x_down_10 = self.doubleConv_down_8(x_down_9 + ring(x_down_2) + cutting(x_up_9, x_down_9.shape[2]))
        x_up_10 = self.dilationConv_up_8(x_up_9 + padding(x_down_9, x_up_9.shape[2]) + ring(x_up_2) + self.LGE_8(e1))

        x_skip = torch.cat((padding(x_down_10, x_up_10.shape[2]), padding(x_down_10, x_up_10.shape[2]), x_up_10), dim=1)
        x_attn = self.sigmoid(self.conv(x_up_10))
        x = torch.mul(x_attn, x_skip)
        x = x + x_skip
        x = self.finalConv(x)

        return x


if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)
    model = DM_DICNet()
    out = model(x)
    print(out.shape)
    from thop import profile

    flops, params = profile(model, inputs=(x,))
    print(f'Flops: {flops}, params: {params}')