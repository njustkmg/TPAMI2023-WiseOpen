'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging
import time
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage.filters import threshold_otsu

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter',
           'accuracy_open', 'ova_loss', 'compute_roc',
           'compute_roc_aupr', 'misc_id_ood', 'ova_ent',
           'test_ood', 'test', 'test_upper', 'exclude_dataset_via_var', 
           'DistAlignQueue', 'cal_Lu', 'exclude_dataset_via_loss']


def ce_loss(logits, targets, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

def consistency_loss(logits, targets, name='ce', mask=None, return_list=False):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None and not return_list:
        # mask must not be boolean type
        loss = loss * mask

    if not return_list:
        loss = loss.mean()

    return loss


class DistAlignQueue(object):
    """
    Distribution Alignment Hook for conducting distribution alignment
    """
    def __init__(self, num_classes, queue_length=128, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.queue_length = queue_length

        # p_target
        self.p_target_ptr, self.p_target = self.set_p_target(p_target_type, p_target)    
        print('distribution alignment p_target:', self.p_target.mean(dim=0))
        # p_model
        self.p_model = torch.zeros(self.queue_length, self.num_classes, dtype=torch.float)
        self.p_model_ptr = torch.zeros(1, dtype=torch.long)

    @torch.no_grad()
    def dist_align(self, probs_x_ulb, probs_x_lb=None, update=True):
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)
            if self.p_target_ptr is not None:
                self.p_target_ptr = self.p_target_ptr.to(probs_x_ulb.device)
        
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(probs_x_ulb.device)
            self.p_model_ptr = self.p_model_ptr.to(probs_x_ulb.device)
        # update queue
        if update:
            self.update_p(probs_x_ulb, probs_x_lb)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target.mean(dim=0) + 1e-6) / (self.p_model.mean(dim=0) + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned
    
    @torch.no_grad()
    def update_p(self, probs_x_ulb, probs_x_lb):
        probs_x_ulb = probs_x_ulb.detach()
        p_model_ptr = int(self.p_model_ptr)
        self.p_model[p_model_ptr] = probs_x_ulb.mean(dim=0)
        self.p_model_ptr[0] = (p_model_ptr + 1) % self.queue_length

        if self.p_target_ptr is not None:
            assert probs_x_lb is not None
            p_target_ptr = int(self.p_target_ptr)
            self.p_target[p_target_ptr] = probs_x_lb.mean(dim=0)
            self.p_target_ptr[0] = (p_target_ptr + 1) % self.queue_length
    
    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        p_target_ptr = None
        if p_target_type == 'uniform':
            p_target = torch.ones(self.queue_length, self.num_classes, dtype=torch.float) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.zeros((self.queue_length, self.num_classes), dtype=torch.float)
            p_target_ptr = torch.zeros(1, dtype=torch.long)
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)
            p_target = p_target.unsqueeze(0).repeat((self.queue_length, 1))
        
        return p_target_ptr, p_target

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_open(pred, target, topk=(1,), num_classes=5):
    """Computes the precision@k for the specified values of k,
    num_classes are the number of known classes.
    This function returns overall accuracy,
    accuracy to reject unknown samples,
    the size of unknown samples in this batch."""
    maxk = max(topk)
    batch_size = target.size(0)
    pred = pred.view(-1, 1)
    pred = pred.t()
    ind = (target == num_classes)
    unknown_size = len(ind)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if ind.sum() > 0:
        unk_corr = pred.eq(target).view(-1)[ind]
        acc = torch.sum(unk_corr).item() / unk_corr.size(0)
    else:
        acc = 0

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], acc, unknown_size


def compute_roc(unk_all, label_all, num_known):
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all)

def compute_roc_aupr(unk_all, label_all, num_known):
    '''
    roc: 以ood数据为positive
    aupr_out：以ood为positive
    aupr_in：以in为positive
    '''
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all), \
           average_precision_score(Y_test, unk_all), \
           average_precision_score(1 - Y_test, -1.0 * unk_all)


def misc_id_ood(score_id, score_ood):
    '''
    roc: 以ood数据为positive
    aupr_out：以ood为positive
    aupr_in：以id为positive
    '''
    id_all = np.r_[score_id, score_ood]
    Y_test = np.zeros(score_id.shape[0]+score_ood.shape[0])
    Y_test[score_id.shape[0]:] = 1
    return roc_auc_score(Y_test, id_all),\
           average_precision_score(Y_test, id_all), \
           average_precision_score(1 - Y_test, -1.0 * id_all)


def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    return Le


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_var(pp, grads_l, grad_dims):
    cnt = 0
    ret = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            ret += torch.sum((grads_l[beg: en] - param.grad.data.view(-1)) ** 2)
            # grads_l[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return ret

def store_grad(pp, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads[:].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def get_targets_q(args, logits_x_ulb, logits_mb_x_ulb, num_ulb, dist_align, update):
    p = F.softmax(logits_x_ulb, dim=-1)
    if not args.p_grad:
        targets_p = p.detach()
    else:
        targets_p = p
    if args.dist_align:
        targets_p = dist_align.dist_align(probs_x_ulb=targets_p, update=update)

    logits_mb = logits_mb_x_ulb.view(num_ulb, 2, -1)
    r = F.softmax(logits_mb, 1)
    tmp_range = torch.arange(0, num_ulb).long().to(args.device)

    o_neg = r[tmp_range, 0, :]
    o_pos = r[tmp_range, 1, :]
    q = torch.zeros((num_ulb, args.num_classes + 1)).to(args.device)
    q[:, :args.num_classes] = targets_p * o_pos
    q[:, args.num_classes] = torch.sum(targets_p * o_neg, 1)
    return q


def cal_Lu(args, logits, logits_open, logits_mb, num_ulb, dist_align, update=True, return_list=False):
    logits_x_ulb_w, logits_x_ulb_s = logits.chunk(2)
    logits_open_x_ulb_w, logits_open_x_ulb_s = logits_open.chunk(2)
    logits_mb_x_ulb_w, logits_mb_x_ulb_s = logits_mb.chunk(2)

    with torch.no_grad():
        p = F.softmax(logits_x_ulb_w, dim=-1)
        targets_p = p.detach()
        if args.dist_align:
            targets_p = dist_align.dist_align(probs_x_ulb=targets_p, update=update)

        logits_mb = logits_mb_x_ulb_w.view(num_ulb, 2, -1)
        r = F.softmax(logits_mb, 1)
        tmp_range = torch.arange(0, num_ulb).long().to(args.device)
        out_scores = torch.sum(targets_p * r[tmp_range, 0, :], 1)
        in_mask = (out_scores < 0.5)

        o_neg = r[tmp_range, 0, :]
        o_pos = r[tmp_range, 1, :]
        q = torch.zeros((num_ulb, args.num_classes + 1)).to(args.device)
        q[:, :args.num_classes] = targets_p * o_pos
        q[:, args.num_classes] = torch.sum(targets_p * o_neg, 1)
        targets_q = q.detach()

    p_probs_x_ulb = targets_p.detach()
    p_max_probs, p_max_idx = torch.max(p_probs_x_ulb, dim=-1)
    if args.mb_mask:
        conf = r[tmp_range, 1, p_max_idx]
        p_mask = conf.ge(args.p_cutoff).to(p_max_probs.dtype)
    else:
        p_mask = p_max_probs.ge(args.p_cutoff).to(p_max_probs.dtype)

    q_probs_x_ulb = targets_q.detach()
    q_max_probs, _ = torch.max(q_probs_x_ulb, dim=-1)
    q_mask = q_max_probs.ge(args.q_cutoff).to(q_max_probs.dtype)

    L_ui = consistency_loss(logits_x_ulb_s, targets_p, 'ce', mask=in_mask * p_mask, return_list=return_list)
    if args.mb_op:
        tmp = get_targets_q(args, logits_x_ulb_w, logits_mb_x_ulb_s, num_ulb, dist_align, update)
        L_op = consistency_loss(tmp, targets_q, 'ce', mask=q_mask, return_list=return_list)
    else:
        L_op = consistency_loss(logits_open_x_ulb_s, targets_q, 'ce', mask=q_mask, return_list=return_list)

    return L_ui, L_op, p_mask, q_mask

def exclude_dataset_via_loss(args, labeled_trainloader, grad_dims, dataset, model, dist_align):
    data_time = AverageMeter()
    end = time.time()
    dataset.init_index()
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0], ncols=80, ascii=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, ((inputs_u_w, inputs_u_s), target_god_all) in enumerate(test_loader):
            num_ulb = inputs_u_w.shape[0]
            inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(args.device)
            logits, logits_mb, logits_open = model(inputs_ws)

            L_ui, L_op, _, _ = cal_Lu(args, logits, logits_open, logits_mb, num_ulb, dist_align, update=False, return_list=True)
            loss = L_ui + L_op

            if batch_idx == 0:
                losses = loss
                labels = target_god_all
            else:
                losses = torch.cat([losses, loss], 0)   
                labels = torch.cat([labels, target_god_all], 0) 

        if not args.no_progress:
            test_loader.close()

    k = int(len(losses) * 0.10)
    values, indices = torch.topk(losses, k)
    th_90 = values[-1]
    print("top-90%:", th_90)
    k = int(len(losses) * 0.05)
    values, indices = torch.topk(losses, k)
    th_95 = values[-1]
    print("top-95%:", th_95)

    if args.loss_topk:
        if args.loss_topk_ratio != 0:
            k = int(len(losses) * args.loss_topk_ratio)
            values, indices = torch.topk(losses, k)
            th = values[-1]
        else:
            th = float('inf')
    else:
        th = threshold_otsu(losses.cpu().numpy().reshape(-1,1))
  
    selected_mask_df = losses < th
    all_selected = np.where(selected_mask_df.data.cpu().numpy() != 0)[0]

    model.train()
    return all_selected, selected_mask_df, losses.tolist(), labels, th

def exclude_dataset_via_var(args, labeled_trainloader, grad_dims, dataset, model, dist_align):
    loss_list = []
    model.eval()
    dataset.init_index()
    print('')
    print('='*10, 'start cal grads_l_d', '='*10)
    print('')
    l_loader = tqdm(labeled_trainloader, disable=args.local_rank not in [-1, 0], ncols=80, ascii=True) 
    for batch_idx, (inputs_x, targets_x) in enumerate(l_loader):
        inputs = inputs_x.to(args.device)
        targets_x = targets_x.to(args.device)
        logits, logits_mb, logits_open = model(inputs)
        ## Loss for labeled samples
        L_x = F.cross_entropy(logits, targets_x, reduction='mean')
        L_mb = ova_loss(logits_mb, targets_x)
        loss_l = L_x + L_mb
        loss_l.backward()
        if batch_idx == 0:
            grads_l = (torch.Tensor(sum(grad_dims))).to(args.device)
            store_grad(model.parameters, grads_l, grad_dims)
        else:
            tmp = (torch.Tensor(sum(grad_dims))).to(args.device)
            store_grad(model.parameters, tmp, grad_dims)
            grads_l.add_(tmp)
        model.zero_grad()
    if not args.no_progress:
            l_loader.close()

    grads_l = grads_l / len(labeled_trainloader.dataset) * args.batch_size

    var_list = []
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0], ncols=80, ascii=True)
    for batch_idx, ((inputs_u_w, inputs_u_s), target_god_all) in enumerate(test_loader):
        num_ulb = inputs_u_w.shape[0]
        inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(args.device)
        logits, logits_mb, logits_open = model(inputs_ws)

        L_ui, L_op, _, _ = cal_Lu(args, logits, logits_open, logits_mb, num_ulb, dist_align, update=False)
        loss = L_ui + L_op

        
        loss.backward()
        var_list.append(cal_var(model.parameters, grads_l, grad_dims).item())
        model.zero_grad()

        loss_list.append(loss.item())
        if batch_idx == 0:
            labels = target_god_all
        else:
            labels = torch.cat([labels, target_god_all], 0)
    if not args.no_progress:
        test_loader.close()

    var_all = torch.tensor(var_list).to(args.device)
    var_np = np.array(var_list)
    if args.var_topk:
        k = int(len(var_all) * args.var_topk_ratio)
        if k==0:
            th = float('inf')
        else:
            values, _ = torch.topk(var_all, k)
            th = values[-1]
    else:
        th = threshold_otsu(var_np.reshape(-1,1))
    selected_mask_df = var_all <= th

    all_selected = np.where(selected_mask_df.data.cpu().numpy() != 0)[0]
    model.train()
    return all_selected, selected_mask_df, var_all, loss_list, labels, th



def test_upper(args, test_loader, model, epoch, val=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_known = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0], ncols=80, ascii=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs, outputs_open, _ = model(inputs)
            outputs = F.softmax(outputs, 1)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, min(5,args.num_classes)))
            top1.update(prec1.item(), outputs.shape[0])
            top5.update(prec5.item(), outputs.shape[0])
            known_targets = targets < args.num_classes - 1
            # known_targets = targets < 6#[0]
            known_pred = outputs[known_targets]
            known_targets = targets[known_targets]

            if len(known_pred) > 0:
                prec1, _ = accuracy(known_pred, known_targets, topk=(1, min(5,args.num_classes)))
                top1_known.update(prec1.item(), known_pred.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
                                            "Data: {data:.3f}s."
                                            "Batch: {bt:.3f}s. "
                                            "Loss: {loss:.4f}. "
                                            "Closed t1: {top1:.3f} "
                                            "t5: {top5:.3f} ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    if not val:
        logger.info("Closed acc: {:.3f}".format(top1_known.avg))
        logger.info("Overall acc: {:.3f}".format(top1.avg))
        return losses.avg, top1.avg, top1_known.avg
    else:
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        return top1.avg

def test(args, test_loader, model, epoch, val=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    acc = AverageMeter()
    unk = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0], ncols=80, ascii=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs, outputs_open, _ = model(inputs)
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, out_open.size(0)).long().cuda(args.device)
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            known_score = outputs.max(1)[0]
            targets_unk = targets >= int(outputs.size(1))
            targets[targets_unk] = int(outputs.size(1))
            known_targets = targets < int(outputs.size(1))#[0]
            # known_targets = targets < 6#[0]
            known_pred = outputs[known_targets]
            known_targets = targets[known_targets]

            if len(known_pred) > 0:
                prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, min(5,args.num_classes)))
                top1.update(prec1.item(), known_pred.shape[0])
                top5.update(prec5.item(), known_pred.shape[0])

            ind_unk = unk_score > 0.5
            pred_close[ind_unk] = int(outputs.size(1))
            acc_all, unk_acc, size_unk = accuracy_open(pred_close,
                                                       targets,
                                                       num_classes=int(outputs.size(1)))
            acc.update(acc_all.item(), inputs.shape[0])
            unk.update(unk_acc, size_unk)

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
                known_all = known_score
                label_all = targets
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, targets], 0)

            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
                                            "Data: {data:.3f}s."
                                            "Batch: {bt:.3f}s. "
                                            "Loss: {loss:.4f}. "
                                            "Closed t1: {top1:.3f} "
                                            "t5: {top5:.3f} "
                                            "acc: {acc:.3f}. "
                                            "unk: {unk:.3f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    acc=acc.avg,
                    unk=unk.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    ## ROC calculation
    #import pdb
    #pdb.set_trace()
    unk_all = unk_all.data.cpu().numpy()
    known_all = known_all.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()
    if not val:
        roc, aupr_out, aupr_in= compute_roc_aupr(unk_all, label_all,
                          num_known=int(outputs.size(1)))
        roc_soft = compute_roc(-known_all, label_all,
                               num_known=int(outputs.size(1)))
        ind_known = np.where(label_all < int(outputs.size(1)))[0]
        id_score = unk_all[ind_known]
        ## 2023.4.4
        # id_score = -known_all[ind_known]
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        logger.info("Overall acc: {:.3f}".format(acc.avg))
        logger.info("Unk acc: {:.3f}".format(unk.avg))
        logger.info("ROC: {:.3f}".format(roc))
        logger.info("ROC Softmax: {:.3f}".format(roc_soft))
        logger.info("AUPR(in): {:.3f}".format(aupr_in))
        logger.info("AUPR(out): {:.3f}".format(aupr_out))
        return losses.avg, top1.avg, acc.avg, \
               unk.avg, roc, roc_soft, id_score, \
               aupr_in, aupr_out
    else:
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        return top1.avg


def test_ood(args, test_id, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0], ncols=80, ascii=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            outputs, outputs_open, _ = model(inputs)
            outputs = F.softmax(outputs, 1)
            known_score = outputs.max(1)[0]
            # out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            # tmp_range = torch.arange(0, out_open.size(0)).long().cuda(args.device)
            # pred_close = outputs.data.max(1)[1]
            # unk_score = out_open[tmp_range, 0, pred_close]
            unk_score = -known_score
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
        if not args.no_progress:
            test_loader.close()
    ## ROC calculation
    unk_all = unk_all.data.cpu().numpy()
    roc, aupr_out, aupr_in = misc_id_ood(test_id, unk_all)

    return roc, aupr_in, aupr_out
