import numpy as np
import torch.nn as nn
import scipy.sparse as sp
# from sklearn.neighbors import KNeighborsClassifier


import torchvision.transforms as transforms
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))  # 每行之和形成ndarray
        r_inv = np.power(rowsum, -1).flatten()  # 前者做倒数后折叠成一维数组(1*1)
        r_inv[np.isinf(r_inv)] = 0.  # 将除0元素置0，np的布尔索引技术
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)  # mx转置的标准化
        return mx

    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 稀疏矩阵以行-列-数据三表存储
    indices = torch.from_numpy(np.vstack(
        (sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_matrix(nc):
    adj = sp.coo_matrix((np.ones(nc), (range(nc), range(nc))), shape=(nc, nc), dtype='float32')  # 对角矩阵生成
    adj = normt_spm(adj, method='in')  # 矩阵转置并正规化
    adj = spm_to_tensor(adj)  # 转为稀疏矩阵张量
    return adj


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def get_RANK(query_semantic, test_mask, classes):
    query_semantic = query_semantic.cpu().numpy()
    test_mask = test_mask.cpu().numpy()
    query_semantic = query_semantic / np.linalg.norm(query_semantic, 2, axis=1, keepdims=True)
    test_mask = test_mask / np.linalg.norm(test_mask, 2, axis=1, keepdims=True)
    dist = np.dot(query_semantic, test_mask.transpose())
    return classes[np.argmax(dist, axis=1)]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return softmax_x


def compute_domain_accuracy(predict_label, domain):
    num = predict_label.shape[0]
    n = 0
    for i in predict_label:
        if i in domain:
            n += 1

    return float(n) / num


def compute_class_accuracy_total(true_label, predict_label, classes):
    nclass = len(classes)
    acc_per_class = np.zeros((nclass, 1))
    for i, class_i in enumerate(classes):
        idx = np.where(true_label == class_i)[0]
        acc_per_class[i] = (sum(true_label[idx] == predict_label[idx]) * 1.0 / len(idx))
    return np.mean(acc_per_class)


def entropy(probs):
    """ Computes entropy. """
    max_score = np.max(probs, axis=1)
    return -max_score * np.log(max_score)


def opt_domain_acc(cls_s, cls_t):
    ''' source domain '''
    opt_acc_s = 0
    num_s = cls_s.shape[0]
    max_score_s = np.max(cls_s, axis=1)

    opt_acc_t = 0
    num_t = cls_t.shape[0]
    max_score_t = np.max(cls_t, axis=1)

    max_H = 0
    opt_tau = 0
    for step in range(10):
        tau = 0.1 * step

        idx = np.where(max_score_s > tau)
        acc_s = float(idx[0].shape[0]) / num_s

        idx = np.where(max_score_t < tau)
        acc_t = float(idx[0].shape[0]) / num_t

        H = 2 * acc_s * acc_t / (acc_s + acc_t)
        if H > max_H:
            opt_acc_t = acc_t
            opt_acc_s = acc_s
            max_H = H
            opt_tau = tau
    return opt_acc_s, opt_acc_t, opt_tau


def post_process(v_prob, a_prob, gt, split_num, seen_c, unseen_c, data):
    v_max = np.max(v_prob, axis=1)
    H_v = entropy(v_prob)
    v_pre = np.argmax(v_prob, axis=1)

    a_max = np.max(v_prob, axis=1)
    H_a = entropy(a_prob)
    a_pre = np.argmax(a_prob, axis=1)

    opt_S = 0
    opt_U = 0
    opt_H = 0
    opt_Ds = 0
    opt_Du = 0
    opt_tau = 0

    for step in range(9):
        base = 0.1 * step + 0.1
        tau = -base * np.log(base)
        pre = v_pre
        for idx, class_i in enumerate(pre):
            if v_max[idx] - base < 0:
                pre[idx] = a_pre[idx]

        pre_s = pre[:split_num]
        pre_t = pre[split_num:]
        gt_s = gt[:split_num]
        gt_t = gt[split_num:]
        S = compute_class_accuracy_total(gt_s, pre_s, seen_c)
        U = compute_class_accuracy_total(gt_t, pre_t, unseen_c)
        Ds = compute_domain_accuracy(pre_s, seen_c)
        Du = compute_domain_accuracy(pre_t, unseen_c)
        H = 2 * S * U / (S + U)

        print('S: {:.4f} U {:.4f} H {:.4f} Ds {:.4f} Du_{:.4f} tau {:.4f}'.format(S, U, H, Ds, Du, base))

        if H > opt_H:
            opt_S = S
            opt_U = U
            opt_H = H
            opt_Ds = Ds
            opt_Du = Du
            opt_tau = tau

    return opt_H, opt_S, opt_U, opt_Ds, opt_Du, opt_tau


def preprocess_strategy(dataset, args):
    evaluate_transforms = None
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    if args.flippingtest:
        val_transforms = transforms.Compose([
            transforms.Resize(480),
            transforms.CenterCrop(448),
            transforms.Lambda(lambda x: [x, transforms.RandomHorizontalFlip(p=1.0)(x)]),
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            transforms.Lambda(lambda crops: [normalize(crop) for crop in crops]),
            transforms.Lambda(lambda crops: torch.stack(crops))
        ])
    else:
        val_transforms = transforms.Compose([
            transforms.Resize(480),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])

    return train_transforms, val_transforms


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, optimizer1, optimizer2, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr1 * (0.1 ** (epoch // args.epoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    lr = args.lr1 * (0.1 ** (epoch // args.epoch_decay))
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr

    lr = args.lr2 * (0.1 ** (epoch // args.epoch_decay))
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
