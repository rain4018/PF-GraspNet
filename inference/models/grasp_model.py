import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.feature import peak_local_max


def get_PFL(pos_pred, sin_pred, cos_pred, width_pred,
        sin_gt, cos_gt, width_gt, min_distance=3, threshold=0.2):

    pos_map = pos_pred[0, 0].detach().cpu().numpy()  # [H, W]
    local_max = peak_local_max(pos_map, min_distance=min_distance, threshold_abs=threshold,
                               num_peaks=50)

    if len(local_max) == 0:
        return torch.tensor(0.0, device=pos_pred.device)

    coords = torch.tensor(local_max, dtype=torch.long).to(pos_pred.device)
    y, x = coords[:, 0], coords[:, 1]

    sin_pred_vals = sin_pred[0, 0, y, x]
    cos_pred_vals = cos_pred[0, 0, y, x]
    width_pred_vals = width_pred[0, 0, y, x]

    sin_gt_vals = sin_gt[0, 0, y, x]
    cos_gt_vals = cos_gt[0, 0, y, x]
    width_gt_vals = width_gt[0, 0, y, x]

    PFL = (
            F.smooth_l1_loss(sin_pred_vals, sin_gt_vals) +
            F.smooth_l1_loss(cos_pred_vals, cos_gt_vals) +
            F.smooth_l1_loss(width_pred_vals, width_gt_vals)
    )
    return PFL


class GraspModel(nn.Module):
    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        PFL = get_PFL(pos_pred, sin_pred, cos_pred, width_pred, y_sin, y_cos, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss + 0.4 * PFL,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss,
                'Peak-Focused_loss': 0.4 * PFL
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }


class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
