import torch
def make_one_hot(labels,classes,rep=1):
    # print(labels.size())
    
    n = labels.size()[0]
    one_hot = torch.FloatTensor(n, classes).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = target.unsqueeze(2).repeat(1,1,rep).view(target.size(0),-1)
    return target


def cross_entropy_with_mask(logits, targets, labels, num_classes, K):
    # idx = targets!=-100

    # logits = logits[idx]
    # targets = targets[idx]
    # labels = labels[idx]
    targets = targets.view(-1, 1)
    labels = labels.view(-1, 1)
    target_mask = make_one_hot(targets,K*num_classes,1)
    sub_cls = K
    positive_mask = make_one_hot(labels,num_classes,sub_cls)

    # target_mask, positive_mask
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    negative_mask = 1 - positive_mask
    denominator_mask = negative_mask + target_mask
    
    exp_logits = torch.exp(logits) * denominator_mask
    denominator = exp_logits.sum(1, keepdim=True)
    log_prob = logits - torch.log(denominator)
    
    mean_log_prob_pos = (target_mask * log_prob).sum(1)
    loss = (-mean_log_prob_pos).mean()
    
    return loss

