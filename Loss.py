import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt as edt


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.detach().cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha
        distance = distance.to(pred_error.device)

        dt_field = pred_error * distance
        loss = dt_field.mean()
        return loss


class MultiClassHausdorffDTLoss(nn.Module):
    """Multi-class Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0):
        super(MultiClassHausdorffDTLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (b, C, x, y, z) or (b, C, x, y)
        target: (b, C, x, y, z) or (b, C, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert pred.dim() == target.dim(), "Prediction and target need to be of same dimension"
        assert pred.size(1) == target.size(1), "Number of channels in prediction should match num_classes"

        C = target.shape[1]
        hd_loss = HausdorffDTLoss()
        total_loss = 0
        pred = F.softmax(pred, dim=1)  # Applying softmax to the prediction

        for i in range(C):
            hd_loss_i = hd_loss(pred[:, i], target[:, i])
            total_loss += hd_loss_i

        return total_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0
        input = F.softmax(input, dim=1)
        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


class SurfaceLoss(nn.Module):
    def __init__(self, idc):
        super(SurfaceLoss, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = idc

    def forward(self, probs, dist_maps):
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        loss_over_idc = torch.einsum("bcdwh,bcdwh->bcdwh", pc, dc)      # 点乘
        surface_loss = torch.mean(loss_over_idc)
        return surface_loss
