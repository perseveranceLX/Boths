import torch.nn as nn
import pywt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

w = pywt.Wavelet('db1')

dec_hi = torch.Tensor(w.dec_hi[::-1])
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

Lfilters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)], dim=0)
Mfilters = torch.stack([dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1), dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)], dim=0)
Hfilters = torch.stack([dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)


def dwt(img):
    Lfilters_cat = torch.cat(tuple(Lfilters[:, None]) * img.shape[1], 0)
    Lfilters_cat = Lfilters_cat.unsqueeze(1)
    Mfilters_cat = torch.cat(tuple(Mfilters[:, None]) * img.shape[1], 0)
    Mfilters_cat = Mfilters_cat.unsqueeze(1)
    Hfilters_cat = torch.cat(tuple(Hfilters[:, None]) * img.shape[1], 0)
    Hfilters_cat = Hfilters_cat.unsqueeze(1)
    return F.conv2d(img, Variable(Lfilters_cat.cuda(), requires_grad=True), stride=2, groups=img.shape[1]) \
        , F.conv2d(img, Variable(Mfilters_cat.cuda(), requires_grad=True), stride=2, groups=img.shape[1]) \
        , F.conv2d(img, Variable(Hfilters_cat.cuda(), requires_grad=True), stride=2, groups=img.shape[1])


MSE_Loss = nn.MSELoss()


class WMSEloss(nn.Module):
    def __init__(self):
        super(WMSEloss, self).__init__()

    def forward(self, x, y, r=0.5):
        loss = 0
        loss += MSE_Loss(x, y)
        l, m, h = 1, 1, 1
        for i in range(2):
            l, m, h = l * r * r, l * r * (1 - r), l * (1 - r) * (1 - r)
            x0, x1, x2 = dwt(x)
            y0, y1, y2 = dwt(y)
            loss = loss + MSE_Loss(x1, y1) * 2 * m + MSE_Loss(x2, y2) * h
            x, y = x0, y0
        loss += MSE_Loss(x0, y0) * l

        return loss
