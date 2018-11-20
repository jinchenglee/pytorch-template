import torch


def precision(y_pred, y_true, threshold=0.5, eps=1e-9):
    with torch.no_grad():
        y_pred = torch.ge(y_pred.float(), threshold).float()
        y_true = y_true.float()

        # 'dim=0' reduce along class dimension
        # 'dim=1' reduce along per-sample dimension
        true_positive = (y_pred * y_true).sum(dim=0)
        prec = true_positive.div(y_pred.sum(dim=0).add(eps))
    return prec

def recall(y_pred, y_true, threshold=0.5, eps=1e-9):
    with torch.no_grad():
        y_pred = torch.ge(y_pred.float(), threshold).float()
        y_true = y_true.float()

        # 'dim=0' reduce along class dimension
        # 'dim=1' reduce along per-sample dimension
        true_positive = (y_pred * y_true).sum(dim=0)
        precision = true_positive.div(y_pred.sum(dim=0).add(eps))
        rec = true_positive.div(y_true.sum(dim=0).add(eps))    
    return rec

def fbeta_score(y_pred, y_true, beta=1, threshold=0.5, eps=1e-9):
    with torch.no_grad():
        beta2 = beta**2

        y_pred = torch.ge(y_pred.float(), threshold).float()
        y_true = y_true.float()

        # 'dim=0' reduce along class dimension
        # 'dim=1' reduce along per-sample dimension
        true_positive = (y_pred * y_true).sum(dim=0)
        precision = true_positive.div(y_pred.sum(dim=0).add(eps))
        recall = true_positive.div(y_true.sum(dim=0).add(eps))
        f_score = (precision*recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2)

    return f_score
