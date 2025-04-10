import torch
import torch.nn as nn
import torchvision


def to_inputsize(pred_masks, true_masks, dataset_name):
    assert dataset_name in ['CASIA-ALL', 'MICHE', 'CASIA-Iris-M1', 'Lamp', 'Thousand']
    if dataset_name == 'CASIA-Iris-M1':
        tH, tW = 400, 400
    else:
        tH, tW = 480, 640

    h, w = true_masks.shape[2], true_masks.shape[3]
    if h>tH and w>tW:
        crop_to = torchvision.transforms.CenterCrop((tH, tW))
        true_masks = crop_to(true_masks)
        pred_masks = crop_to(pred_masks)
    if h<tH and w<tW:
        pad_to = nn.ZeroPad2d(((tH-h)//2, (tH-h)//2, (tW-w)//2, (tW-w)//2))
        true_masks = pad_to(true_masks)
        pred_masks = pad_to(pred_masks)
    return pred_masks, true_masks
  
def compute_tfpn(true_mask, pred_mask):
    '''
    return: dict {tp, fp, tn, fn}
    input shape: cwh = 1x400x400
    '''
    c, r = true_mask.shape[1], pred_mask.shape[2]
    num_pixel = c*r

    # to BoolTensor
    true_mask = true_mask>0
    pred_mask = pred_mask>0
    
    tp = (true_mask & pred_mask).sum()
    fp = (~true_mask & pred_mask).sum()
    tn = (~(true_mask | pred_mask)).sum()
    fn = (true_mask & (~pred_mask)).sum()

    return {
        'tp': tp/num_pixel,
        'fp': fp/num_pixel,
        'tn': tn/num_pixel,        
        'fn': fn/num_pixel
    }

def compute_e1(n_batch, true_masks, pred_masks):
    '''
    return the middle e1, to get final e1 as below:
    e1 += compute_e1(true_masks, mask_pred)
    e1 = torch.div(e1, n_val)
    e1 = min(e1, 1-e1)*100
    '''
    sum_e1 = 0
    for i in range(n_batch):
        tpfn = compute_tfpn(true_masks[i], pred_masks[i])
        fp, fn = tpfn['fp'], tpfn['fn']
        sum_e1 += fp+fn
    return sum_e1/n_batch


def compute_miou(n_batch, true_masks, pred_masks):
    sum_iou = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(true_masks[i], pred_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if tp+fn+fp == 0:
            iou=1
        else:
            iou=tp/(tp+fn+fp)
        sum_iou += iou
    return sum_iou/n_batch

def compute_dice(n_batch, true_masks, pred_masks):
    sum_dice = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(true_masks[i], pred_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if 2*tp+fn+fp == 0:
            dice=1
        else:
            dice=2*tp/(2*tp+fn+fp)
        sum_dice += dice
    return sum_dice/n_batch

def compute_recall(n_batch, true_masks, pred_masks):
    sum_rc = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(true_masks[i], pred_masks[i])
        tp, fn = tfpn['tp'], tfpn['fn']
        sum_rc += tp / (tp+fn)
    return sum_rc/n_batch

def compute_precision(n_batch, true_masks, pred_masks):
    sum_p = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(true_masks[i], pred_masks[i])
        tp, fp = tfpn['tp'], tfpn['fp']
        sum_p += tp / (tp+fp+(1e-10))
    return sum_p/n_batch

def compute_f1(n_batch, true_masks, pred_masks):
    sum_f1 = 0
    for i in range(n_batch):
        tfpn = compute_tfpn(true_masks[i], pred_masks[i])
        tp, fp, fn = tfpn['tp'], tfpn['fp'], tfpn['fn']
        if tp+fp == 0:
            precision = tp
        else:
            precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        if precision+recall == 0:
            f1 = tp
        else:
            f1 = (2*precision*recall) / (precision+recall)
        if f1 > 999:
            f1 = 0
        sum_f1 += f1
    return sum_f1/n_batch

def evaluate_iris(pred_masks, true_masks, dataset_name):
    """Evaluation without the densecrf with the dice coefficient"""
    n_batch = true_masks.size()[0]
    e1, iou, dice, f1, recall = 0, 0, 0, 0, 0

    # compute miou, dice, recall, e1, f1
    # pred_masks, true_masks = to_inputsize(pred_masks, true_masks, dataset_name)
    # iou = compute_miou(n_batch, true_masks, pred_masks).item()
    # dice = compute_dice(n_batch, true_masks, pred_masks).item()
    # recall = compute_recall(n_batch, true_masks, pred_masks).item()
    # e1 = compute_e1(n_batch, true_masks, pred_masks).item()
    # f1 = compute_f1(n_batch, true_masks, pred_masks).item()
    iou = compute_miou(n_batch, true_masks, pred_masks)
    dice = compute_dice(n_batch, true_masks, pred_masks)
    recall = compute_recall(n_batch, true_masks, pred_masks)
    e1 = compute_e1(n_batch, true_masks, pred_masks)
    f1 = compute_f1(n_batch, true_masks, pred_masks)
    
    if f1 > 100:
        f1 = 100

    return {
        'E1': e1*100,
        'IoU': iou*100,
        'Dice': dice,
        'F1': f1*100,
        'recall': recall*100
    }
