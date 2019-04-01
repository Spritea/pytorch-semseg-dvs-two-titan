# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        a = n_class * label_true[mask].astype(int)
        b = label_pred[mask]
        tt = a + b
        pp = np.bincount(tt)
        kk = np.bincount(tt, minlength=n_class ** 2)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        lt=label_trues
        lp=label_preds
        t1=lt.flatten()
        t2=lp.flatten()
        fh=self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
        self.confusion_matrix += fh
        # print("kk")

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        k = np.diag(hist)
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        cls_pre = np.diag(hist) / hist.sum(axis=1)
        cls_rec = np.diag(hist) / hist.sum(axis=0)

        cls_f1 = 2 * cls_pre * cls_rec / (cls_pre + cls_rec)
        mean_f1 = np.nanmean(cls_f1)
        return acc,cls_pre,cls_rec,cls_f1, iu, hist

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

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
