import torch
from torchvision.ops import nms

def custom_nms(boxes, scores, iou_threshold):
    """
    Custom Non-Maximum Suppression (NMS) based on intersection over minimum area.

    Args:
    - boxes (Tensor): Bounding boxes of shape (N, 4) representing (x1, y1, x2, y2).
    - scores (Tensor): Confidence scores of shape (N,).
    - iou_threshold (float): Intersection over Minimum Area (IOMA) threshold.

    Returns:
    - keep (Tensor): Indices of the selected boxes after NMS.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # Calculate areas of bounding boxes
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # Sort the boxes by scores in descending order
    _, indices = scores.sort(descending=True)
    boxes = boxes[indices]
    areas = areas[indices]

    keep = []
    while boxes.numel() > 0:
        # Keep the box with the highest score
        keep.append(indices[0].item())

        # Calculate intersection over minimum area (IOMA) with remaining boxes
        xx1 = torch.max(boxes[0, 0], boxes[1:, 0])
        yy1 = torch.max(boxes[0, 1], boxes[1:, 1])
        xx2 = torch.min(boxes[0, 2], boxes[1:, 2])
        yy2 = torch.min(boxes[0, 3], boxes[1:, 3])

        intersection = torch.clamp(xx2 - xx1 + 1, min=0) * torch.clamp(yy2 - yy1 + 1, min=0)
        ioma = intersection / torch.min(areas[0], areas[1:])

        # Keep boxes with IOMA less than the threshold
        mask = ioma <= iou_threshold
        indices = indices[1:][mask]
        boxes = boxes[1:][mask]
        areas = areas[1:][mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)
