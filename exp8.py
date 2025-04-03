# exp 8 accuracy metrics

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
np.random.seed(42)
n = 1000
y_true = np.random.binomial(1, 0.3, n)
y_pred = np.clip(y_true + np.random.normal(0, 0.3, n), 0, 1)
fpr, tpr, thr = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
def metrics_at(th):
    pb = (y_pred >= th).astype(int)
    tp = np.sum((pb==1) & (y_true==1))
    fp = np.sum((pb==1) & (y_true==0))
    tn = np.sum((pb==0) & (y_true==0))
    fn = np.sum((pb==0) & (y_true==1))
    acc = (tp+tn) / n
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    return acc, prec, rec, f1
best_t, best_f1 = 0, 0
for t in thr:
    _, _, _, f1 = metrics_at(t)
    if f1 > best_f1: best_f1, best_t = f1, t
acc, prec, rec, f1 = metrics_at(best_t)
print("ROC AUC: %.3f" % roc_auc)
print("Optimal Threshold: %.3f" % best_t)
print("Acc: %.3f, Prec: %.3f, Rec: %.3f, F1: %.3f" % (acc, prec, rec, f1))
