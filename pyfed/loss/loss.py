import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.hausdorff_loss import HausdorffDTLoss, LogHausdorffDTLoss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, include_background=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred, dim=1)
        num_class = sigmoid_pred.shape[1]
        label = F.one_hot(gt.squeeze(1).long(), num_class).permute(0, 3, 1, 2)
        loss = 0

        for i in range(num_class):
            if i == 0 and not self.include_background:
                continue
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            union = torch.sum(sigmoid_pred[:, i, ...]) + torch.sum(label[:, i, ...])
            loss += (2 * intersect + self.smooth) / (union + self.smooth)
        loss = 1 - loss * 1.0 / (num_class - int(not self.include_background))
        return loss


class DiceLossFundus(nn.Module):
    def __init__(self, smooth=1e-5, include_background=True):
        super(DiceLossFundus, self).__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred, gt):
        pred = F.sigmoid(pred)
        num_class = pred.shape[1]
        loss = 0
        for i in range(num_class):
            if i == 0 and not self.include_background:
                continue
            intersect = torch.sum(pred[:, i, ...] * gt[:, i, ...])
            union = torch.sum(pred[:, i, ...]) + torch.sum(gt[:, i, ...])
            loss += (2 * intersect + self.smooth) / (union + self.smooth)
        loss = 1 - loss * 1.0 / (num_class - int(not self.include_background))
        return loss


class FIMLossFundus(nn.Module):
    def __init__(self, smooth=1e-5, method="w2", conversion_method="2.5D"):
        super(FIMLossFundus, self).__init__()
        self.smooth = smooth
        self.method = method
        self.conversion_method = conversion_method

    def forward(self, pred, gt, img):
        # Convert to L*a*b* manually and take the L channel
        # first convert to XYZ
        img = (
            torch.where(
                img > 0.04045,
                torch.pow((img + 0.055) / 1.055, 2.4),
                img / 12.92,
            )
            * 100
        )
        y = (
            img[:, 0, ...] * 0.2126
            + img[:, 1, ...] * 0.7152
            + img[:, 2, ...] * 0.0722
        )
        # then convert to L*a*b*
        img = (
            116
            * torch.where(
                y / 100 > 0.008856,
                torch.pow(y / 100, 1 / 3),
                7.787 * y / 100 + 16 / 116,
            )
            - 16
        )
        img = img.unsqueeze(1)
        loss = 0.0
        pred = F.sigmoid(pred)
        for class_idx in range(gt.shape[1]):
            fg_mask = (gt[:, class_idx, ...] > 0).float()
            fg_gt_pixels = fg_mask.unsqueeze(1) * img
            fg_pred = pred[:, class_idx, ...]
            fg_pred_pixels = fg_pred.unsqueeze(1) * img
            if self.method == "w2":
                pred_sorted = torch.sort(fg_pred_pixels.flatten(start_dim=1), dim=1)[0]
                gt_sorted = torch.sort(fg_gt_pixels.flatten(start_dim=1), dim=1)[0]
                loss += torch.sqrt(F.mse_loss(pred_sorted, gt_sorted) + self.smooth)
            elif self.method == "mse":
                loss += F.mse_loss(fg_pred_pixels, fg_gt_pixels)
        return loss

class DiceFIMLossFundus(nn.Module):
    def __init__(self, lambda_dice=1, lambda_fim=0.05, smooth=1e-5, fim_method='w2', include_background=True):
        super(DiceFIMLossFundus, self).__init__()
        self.dice = DiceLossFundus(smooth, include_background)
        self.fim = FIMLossFundus(smooth, fim_method)
        self.lambda_dice = lambda_dice
        self.lambda_fim = lambda_fim

    def forward(self, pred, gt, img):
        return self.lambda_dice * self.dice(pred, gt) + self.lambda_fim * self.fim(pred, gt, img)

class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5, include_background=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred, dim=1)
        num_class = sigmoid_pred.shape[1]
        label = F.one_hot(gt.squeeze(1).long(), num_class).permute(0, 3, 1, 2)
        loss = 0

        for i in range(num_class):
            if i == 0 and not self.include_background:
                continue
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            fp = torch.sum(sigmoid_pred[:, i, ...] * (1 - label[:, i, ...]))
            fn = torch.sum((1 - sigmoid_pred[:, i, ...]) * label[:, i, ...])
            loss += (intersect + self.smooth) / (
                intersect + self.alpha * fp + self.beta * fn + self.smooth
            )
        loss = 1 - loss * 1.0 / (num_class - int(not self.include_background))
        return loss


class JointLoss(nn.Module):
    def __init__(
        self, lambda_dice=0.5, lambda_ce=0.5, smooth=1e-5, include_background=True
    ):
        super(JointLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(smooth, include_background)
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, pred, gt):
        ce = self.ce(pred, gt.squeeze(1).long())
        return self.lambda_dice * self.dice(pred, gt) + self.lambda_ce * ce


class TverskyCELoss(nn.Module):
    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        lambda_dice=0.5,
        lambda_ce=0.5,
        smooth=1e-5,
        include_background=True,
    ):
        super(TverskyCELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.tversky = TverskyLoss(alpha, beta, smooth, include_background)
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, pred, gt):
        ce = self.ce(pred, gt.squeeze(1).long())
        return self.lambda_dice * self.tversky(pred, gt) + self.lambda_ce * ce


class FIMLoss(nn.Module):
    def __init__(self, smooth=1e-5, method="w2", conversion_method="2.5D"):
        super(FIMLoss, self).__init__()
        self.smooth = smooth
        self.method = method
        self.conversion_method = conversion_method

    def forward(self, pred, gt, img):
        if img.shape[1] == 3:
            if self.conversion_method == "2.5D":
                # Take the middle slice
                img = img[:, 1, ...].unsqueeze(1)
            elif self.conversion_method == "L*a*b*":
                # Convert to L*a*b* manually and take the L channel
                # first convert to XYZ
                img = (
                    torch.where(
                        img > 0.04045,
                        torch.pow((img + 0.055) / 1.055, 2.4),
                        img / 12.92,
                    )
                    * 100
                )
                y = (
                    img[:, 0, ...] * 0.2126
                    + img[:, 1, ...] * 0.7152
                    + img[:, 2, ...] * 0.0722
                )
                # then convert to L*a*b*
                img = (
                    116
                    * torch.where(
                        y / 100 > 0.008856,
                        torch.pow(y / 100, 1 / 3),
                        7.787 * y / 100 + 16 / 116,
                    )
                    - 16
                )
                img = img.unsqueeze(1)
            else:
                raise ValueError("Conversion method not supported")
        elif img.shape[1] == 1:
            pass
        else:
            raise ValueError("Image shape not supported")

        fg_mask = (gt > 0).float()
        fg_gt_pixels = fg_mask * img
        fg_gt_count = fg_mask.sum(dim=(1, 2, 3))
        pred = F.softmax(pred, dim=1)
        fg_pred = pred[:, 1, ...]  # assuming network has 2 output channels
        fg_pred_pixels = fg_pred.unsqueeze(1) * img

        if self.method == "w2":
            # treat every pixel as sample of a distribution
            # so w2 can be calculated as the root of the mse
            # between the two sorted vectors of samples
            pred_sorted = torch.sort(fg_pred_pixels.flatten(start_dim=1), dim=1)[0]
            gt_sorted = torch.sort(fg_gt_pixels.flatten(start_dim=1), dim=1)[0]
            loss = torch.sqrt(F.mse_loss(pred_sorted, gt_sorted) + self.smooth)
        elif self.method == "mse":
            loss = F.mse_loss(fg_pred_pixels, fg_gt_pixels)
        else:
            raise ValueError("Method not supported")
        return loss


class DiceFIMLoss(nn.Module):
    def __init__(
        self,
        lambda_dice=0.5,
        lambda_fim=0.5,
        smooth=1e-5,
        include_background=True,
        method="w2",
        conversion_method="2.5D",
    ):
        super(DiceFIMLoss, self).__init__()
        self.dice = DiceLoss(smooth, include_background)
        self.fim = FIMLoss(smooth, method, conversion_method)
        self.lambda_dice = lambda_dice
        self.lambda_fim = lambda_fim

    def forward(self, pred, gt, img):
        return self.lambda_dice * self.dice(pred, gt) + self.lambda_fim * self.fim(
            pred, gt, img
        )


class TverskyFIMLoss(nn.Module):
    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        lambda_tversky=0.5,
        lambda_fim=0.5,
        smooth=1e-5,
        include_background=True,
        method="w2",
        conversion_method="2.5D",
    ):
        super(TverskyFIMLoss, self).__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth, include_background)
        self.fim = FIMLoss(smooth, method, conversion_method)
        self.lambda_tversky = lambda_tversky
        self.lambda_fim = lambda_fim

    def forward(self, pred, gt, img):
        return self.lambda_tversky * self.tversky(
            pred, gt
        ) + self.lambda_fim * self.fim(pred, gt, img)


class DiceFIMCELoss(nn.Module):
    def __init__(
        self,
        lambda_dice=0.5,
        lambda_fim=0.04,
        lambda_ce=0.5,
        smooth=1e-5,
        include_background=True,
        method="w2",
        conversion_method="2.5D",
    ):
        super(DiceFIMCELoss, self).__init__()
        self.dice = DiceLoss(smooth, include_background)
        self.fim = FIMLoss(smooth, method, conversion_method)
        self.ce = nn.CrossEntropyLoss()
        self.lambda_dice = lambda_dice
        self.lambda_fim = lambda_fim
        self.lambda_ce = lambda_ce

    def forward(self, pred, gt, img):
        return (
            self.lambda_dice * self.dice(pred, gt)
            + self.lambda_fim * self.fim(pred, gt, img)
            + self.lambda_ce * self.ce(pred, gt.squeeze(1).long())
        )


class TverskyFIMCELoss(nn.Module):
    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        lambda_tversky=0.33,
        lambda_fim=0.33,
        lambda_ce=0.33,
        smooth=1e-5,
        include_background=True,
        method="w2",
        conversion_method="2.5D",
    ):
        super(TverskyFIMCELoss, self).__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth, include_background)
        self.fim = FIMLoss(smooth, method, conversion_method)
        self.ce = nn.CrossEntropyLoss()
        self.lambda_tversky = lambda_tversky
        self.lambda_fim = lambda_fim
        self.lambda_ce = lambda_ce

    def forward(self, pred, gt, img):
        return (
            self.lambda_tversky * self.tversky(pred, gt)
            + self.lambda_fim * self.fim(pred, gt, img)
            + self.lambda_ce * self.ce(pred, gt.squeeze(1).long())
        )


class HausdorffLoss(nn.Module):
    def __init__(self, alpha=2.0, log=True):
        super(HausdorffLoss, self).__init__()
        self.alpha = alpha
        self.loss_fn = (
            LogHausdorffDTLoss(alpha=self.alpha, to_onehot_y=True)
            if log
            else HausdorffDTLoss(alpha=self.alpha, to_onehot_y=True)
        )

    def forward(self, pred, gt):
        return self.loss_fn(pred, gt)


class LogitNormCE(nn.Module):
    def __init__(self, t=1.0):
        super(LogitNormCE, self).__init__()
        self.t = t

    def forward(self, pred, gt):
        logit_norm = torch.nn.functional.normalize(pred, p=2, dim=1) / self.t
        return F.cross_entropy(logit_norm, gt)


class LogitNormDice(nn.Module):
    def __init__(self, smooth=1.0, activation="sigmoid"):
        super(LogitNormDice, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        pred = torch.nn.functional.normalize(pred, p=2, dim=1)
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):
            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred == i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt == i] = 1

            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)

            union = each_pred.view(batch_size, -1).sum(1) + each_gt.view(
                batch_size, -1
            ).sum(1)
            dice = (2.0 * intersection) / (union + 1e-5)

            all_dice += torch.mean(dice)

        return all_dice * 1.0 / num_class

    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred, dim=1)

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]

        bg = torch.zeros_like(gt)
        bg[gt == 0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt == 1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)

        loss = 0
        smooth = 1e-5

        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...])
            y_sum = torch.sum(label[:, i, ...])
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss


class LogitNormJointLoss(nn.Module):
    def __init__(self):
        super(LogitNormJointLoss, self).__init__()
        self.ce = LogitNormCE()
        self.dice = LogitNormDice()

    def forward(self, pred, gt):
        return (self.ce(pred, gt) + self.dice(pred, gt)) / 2


def kl_divergence(alpha):
    shape = list(alpha.shape)
    shape[0] = 1
    ones = torch.ones(tuple(shape)).cuda()

    S = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (alpha - ones).mul(torch.digamma(alpha) - torch.digamma(S)).sum(dim=1, keepdim=True)
    kl = first_term + second_term
    return kl.mean()


class EDL_Dice_Loss(nn.Module):
    def __init__(self, kl_weight=0.01, annealing_step=10):
        super(EDL_Dice_Loss, self).__init__()
        self.smooth = 1e-5
        self.kl_weight = kl_weight
        self.annealing_step = annealing_step

    def forward(self, logit, label, epoch_num):
        K = logit.shape[1]
        logit = torch.clamp_max(logit, 80)
        alpha = torch.exp(logit) + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        pred = alpha / S

        # one_hot_y = torch.zeros(pred.shape).cuda()
        # one_hot_y = one_hot_y.scatter_(1, label, 1.0)
        one_hot_y = F.one_hot(label.squeeze(1).long(), K).permute(0, 3, 1, 2)
        one_hot_y.requires_grad = False

        dice_score = 0
        for class_idx in range(K):
            inter = (pred[:, class_idx, ...] * one_hot_y[:, class_idx, ...]).sum()
            union = (pred[:, class_idx, ...] ** 2).sum() + one_hot_y[:, class_idx, ...].sum()
            dice_score += (2 * inter + self.smooth) / (union + self.smooth)

        dice_score = dice_score / K
        loss_dice = 1 - dice_score

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - one_hot_y) + 1
        loss_kl = annealing_coef * kl_divergence(kl_alpha)

        return loss_dice + self.kl_weight * loss_kl


class EDL_Dice_LossFundus(nn.Module):
    def __init__(self, kl_weight=1e-5, annealing_step=10):
        super(EDL_Dice_LossFundus, self).__init__()
        self.smooth = 1e-5
        self.kl_weight = kl_weight
        self.annealing_step = annealing_step

    def forward(self, logit, label, epoch_num):
        K = logit.shape[1]
        # logit = torch.clamp_max(logit, 80)
        alpha = torch.exp(logit) + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        # pred = alpha / S
        pred = F.sigmoid(logit)
        dice_score = 0
        for class_idx in range(K):
            inter = (pred[:, class_idx, ...] * label[:, class_idx, ...]).sum()
            union = (pred[:, class_idx, ...] ** 2).sum() + label[:, class_idx, ...].sum()
            dice_score += (2 * inter + self.smooth) / (union + self.smooth)

        dice_score = dice_score / K
        loss_dice = 1 - dice_score

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - label) + 1
        loss_kl = annealing_coef * kl_divergence(kl_alpha)

        return loss_dice + self.kl_weight * loss_kl