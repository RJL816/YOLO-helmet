import torch
import torch.nn as nn
import torchvision.ops
import torch.nn.functional as F
from .conv import Conv, autopad

# ==========================================
# 模块 1: Coordinate Attention (位置编码)
# ==========================================
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # [修复] 确保通道数为整数
        inp, oup = int(inp), int(oup)
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out

# ==========================================
# 模块 2: Deformable Conv (动态形变)
# ==========================================
class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        super(DeformConv2d, self).__init__()
        # [修复] 确保通道数为整数
        inc, outc = int(inc), int(outc)
        
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        
        self.conv_offset = nn.Conv2d(inc, 3 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        
        self.weight = nn.Parameter(torch.Tensor(outc, inc, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(outc))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        return torchvision.ops.deform_conv2d(
            input=x, 
            offset=offset, 
            weight=self.weight, 
            bias=self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            mask=mask
        )

# ==========================================
# 模块 3: LG-DCN Block (核心创新)
# ==========================================
class LG_DCN_Block(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        # [修复] 确保通道数为整数
        c1, c2 = int(c1), int(c2)
        
        # 确保通道数是整数
        self.branch_channels = int(c2 // 2)
        
        self.local_branch = nn.Sequential(
            Conv(c1, self.branch_channels, k=3, s=1),
            CoordAtt(self.branch_channels, self.branch_channels) 
        )

        self.dynamic_branch = nn.Sequential(
            DeformConv2d(c1, self.branch_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        self.fusion = Conv(2 * self.branch_channels, c2, k=1) 

        self.add = shortcut and c1 == c2

    def forward(self, x):
        local_feat = self.local_branch(x)
        dynamic_feat = self.dynamic_branch(x)
        
        out = self.fusion(torch.cat((local_feat, dynamic_feat), dim=1))
        return x + out if self.add else out

# ==========================================
# 模块 4: C2f 容器
# ==========================================
class C2f_LG_DCN(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        # [关键修复！！！] 
        # 在这里就把 c1 和 c2 强制转成 int。
        # 无论上层传下来的是 512.0 还是 Tensor(512)，这里都会被清洗干净。
        c1, c2 = int(c1), int(c2)
        
        self.c = int(c2 * e)  # hidden channels
        
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        
        # 这里的计算现在全是 int 了，绝对安全
        self.cv2 = Conv((2 + n) * self.c, c2, 1) 
        
        self.m = nn.ModuleList(LG_DCN_Block(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))