import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaDirectionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, features):
        # features: (T, N, D)
        delta = features[1:] - features[:-1]  # (T-1, N, D)
        return self.head(delta.mean(dim=1))  # (T-1, 2)


def visual_input_to_sample_index(input_ids, image_token_index):
    sample_of = []
    for sample_idx, cur_input_ids in enumerate(input_ids):
        num_images = int((cur_input_ids == image_token_index).sum())
        sample_of.extend([sample_idx] * max(num_images, 1))
    return sample_of


def motion_vector_prediction_loss(preds, direction_gt, direction_gt_mask=None):
    # preds: {sample_idx: (T-1, 2)}, direction_gt: (B, T-1, 2) or (B, 2)
    pred_list, gt_list = [], []
    for sample_idx in sorted(preds):
        if direction_gt_mask is not None and not bool(direction_gt_mask[sample_idx]):
            continue
        pred = preds[sample_idx]
        if pred.shape[0] == 0:
            continue
        gt = direction_gt[sample_idx].to(pred.device)
        if gt.dim() == 1:
            gt = gt.unsqueeze(0).expand(pred.shape[0], -1)
        if gt.shape != pred.shape:
            raise ValueError(f"direction_gt shape {tuple(gt.shape)} != prediction shape {tuple(pred.shape)}")
        pred_list.append(pred)
        gt_list.append(gt)

    if not pred_list:
        return None
    pred, gt = torch.cat(pred_list), torch.cat(gt_list)
    return F.mse_loss(pred.to(gt.dtype), gt)
