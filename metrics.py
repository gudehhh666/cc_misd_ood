import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import sklearn

import pickle



def calc_metrics(loader, model):
    acc, softmax, correct, logit, label = get_metric_values(loader, model)
    # aurc, eaurc
    aurc, eaurc = calc_aurc_eaurc(softmax, correct)
    # fpr, aupr
    auroc, aupr_success, aupr, fpr = calc_fpr_aupr(softmax, correct)
    # new FPR
    new_fpr = cal_new_FPR(correct, softmax)
    # calibration measure ece , mce, rmsce
    ece = calc_ece(softmax, label, bins=15)
    # brier, nll
    nll = calc_nll_brier(softmax, logit, label)
    
    return acc, auroc*100, aupr_success*100, aupr*100, fpr*100, aurc*1000, eaurc*1000, ece*100, nll*10, new_fpr

# AURC, EAURC
def calc_aurc_eaurc(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
    # 展开
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

def cal_new_FPR(y, x):
    # y: labels: 1 for positive 0 for negative
    # x: confidence
    results = dict()
    softmax = np.array(x)
    y = np.array(y)
    x = np.max(softmax, 1)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, x)
    results['AUROC'] = sklearn.metrics.auc(fpr, tpr)

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y, x)
    results['AU_PR_POS'] = sklearn.metrics.auc(recall, precision)
    # print(1-y)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(1 - y, -x)
    results['AU_PR_NEG'] = sklearn.metrics.auc(recall, precision)
    if (x.max() - x.min()) == 0:
        results['FPR-95%-TPR'] = 0
    else:
        for i, delta in enumerate(np.arange(x.min(), x.max(), (x.max() - x.min()) / 10000)):
            tpr = len(x[(y == 1) & (x >= delta)]) / len(x[(y == 1)])
            if 0.9505 >= tpr >= 0.9495:
                fpr = len(x[(1 - y == 1) & (x >= delta)]) / len(x[(1 - y == 1)])
                results['FPR-95%-TPR'] = fpr
                break
            else:
                results['FPR-95%-TPR'] = 0

    for n in results: results[n] = round(100. * results[n], 2)
    print('FPR@TPR95 new {0:.2f}'.format(results['FPR-95%-TPR']))
    return results['FPR-95%-TPR']

# AUPR ERROR
def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    auroc = metrics.auc(fpr, tpr)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]

    precision, recall, thresholds = metrics.precision_recall_curve(correctness, softmax_max)
    aupr_success = metrics.auc(recall, precision)
    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)

    print("AUROC {0:.2f}".format(auroc * 100))
    print('AUPR_Success {0:.2f}'.format(aupr_success * 100))
    print("AUPR_Error {0:.2f}".format(aupr_err*100))
    print('FPR@TPR95 {0:.2f}'.format(fpr_in_tpr_95*100))

    return auroc, aupr_success, aupr_err, fpr_in_tpr_95

# ECE
def calc_ece(softmax, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(np.array(softmax))
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels.long())

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    print("ECE {0:.2f} ".format(ece.item()*100))

    return ece.item()

# NLL & Brier Score
def calc_nll_brier(softmax, logit, label):
    # brier_score = np.mean(np.sum((softmax - label_onehot) ** 2, axis=1))

    logit = torch.tensor(np.array(logit), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    log_softmax = logsoftmax(logit)
    nll = calc_nll(log_softmax, label)
    print("NLL {0:.2f} ".format(nll.item()*10))
    # print('Brier {0:.2f}'.format(brier_score*100))

    return nll.item() #, brier_score

# Calc NLL
def calc_nll(log_softmax, label):
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]

    return -out.sum()/len(out)

# Calc coverage, risk
def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

# Calc aurc, eaurc
def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    print("AURC {0:.2f}".format(aurc*1000))
    print("EAURC {0:.2f}".format(eaurc*1000))

    return aurc, eaurc

# Get softmax, logit
def get_metric_values(loader, model):
    '''
    input:
        loader: test_loader
        model: model
    output:
        total_acc: the total accuracy in this epoch
        list_softmax: the softmax of the model, size = [num_of_loader.data, 10]
        list_correct: the correct of the model, size = [num_of_loader.data, 1]
        list_logit: the naive output of the model, size = [num_of_loader.data, 10]
        
    '''
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        total_acc = 0.
        accuracy = 0.

        list_softmax = []
        list_correct = []
        list_logit = []

        labels_list = []

        for idx, (input, target) in enumerate(loader):
            labels_list.extend(target.cpu().data.numpy())
            input = input.cuda()
            target = target.cuda()
            # print(f"the taget is{target.shape, target}")

            output = model(input)
            # print(f"the output is{output.shape, output}")
            # logit is the naive output of the model, which is a (n * 1class) tensor
            
            # max in PyTorch returns a tuple: 
            #     the first element contains the maximum values, 
            #     and the second element contains the indices of those maximum values. 
            #     Since we're interested in the indices (which represent the predicted classes)
            
            pred = output.data.max(1, keepdim=True)[1]
            # print(f"the pred is{pred.shape, pred}")

            # get the carlibration is the prob
            prob, _pred = F.softmax(output, dim=1).max(1)
            # print(f"the prob is{prob.shape, prob}")
            # print(f"the _pred is{_pred.shape, _pred}")
            

            

            total_acc += pred.eq(target.data.view_as(pred)).sum()

            for i in output:
                list_logit.append(i.cpu().data.numpy())

            # softmax不指定dim，默认操作最后一个dim
            list_softmax.extend(F.softmax(output, dim=1).cpu().data.numpy())
            
            
            for j in range(len(pred)):
                if pred[j] == target[j]:
                    accuracy += 1
                    cor = 1
                else:
                    cor = 0
                list_correct.append(cor)

        total_loss /= len(loader)
        # print(total_acc,  len(loader.dataset))
        total_acc = 100. * total_acc.item() / len(loader.dataset)
        # print(total_acc)

        print('Accuracy {:.2f}'.format(total_acc))
        
        # filename = 'demo.pckl'
        # f = open(filename, 'wb')
        # pickle.dump((output, list_softmax, target), f)
        # print('save the file')
        

    return total_acc, list_softmax, list_correct, list_logit, labels_list



class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


# ece_criterion = ECELoss().cuda()
# nll_criterion = nn.CrossEntropyLoss().cuda()