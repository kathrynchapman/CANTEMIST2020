# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class RankingLoss(nn.Module):
    def __init__(self, doc_batching=False, weights=None):
        super(RankingLoss, self).__init__()
        self.doc_batching = doc_batching
        self.weights = weights

    def forward(self, logits, ranks):
        if self.doc_batching:
            #############################################################
            # 1 - apply sigmoid to logits to get probabilities
            temp = torch.nn.Sigmoid()(logits).cpu()
            ranks = ranks.cpu()
            if self.weights is not None:
                ranks *= self.weights

            n_labels = torch.max(ranks).item()
            if n_labels != 0:
                # 2 - separate true probs from remaining probs; rank true probs (there are no rankings for negative labels)
                true_label_probs = torch.flip(
                    temp[torch.argsort(ranks)[-n_labels:]],
                    dims=(0,))  # these are the first n elements of our new array
                # ^ probabilities of all of the true positive labels, sorted in descending rank order

                remaining_probs = temp[torch.argsort(ranks)[:-n_labels]]
                # ^ probabilities of all of the true negative labels

                # 3 - check if lowest-ranking true positive probability is higher than all negative class probs
                lowest_ranking_prob = true_label_probs[-1]
                highest_remaining_prob = torch.max(remaining_probs)

                temp = lowest_ranking_prob - highest_remaining_prob
                temp[temp < 0] = 0
                add_to_loss1 = torch.sum(temp).item()

                # 4 - check that the probabilities are ranked in order
                vals = self.check_rankings([true_label_probs])
                try:
                    total_correct = sum(sum(t) for t in vals).item()
                    total_labels = sum(len(t) for t in vals)
                    add_to_loss2 = 1 - (total_correct / total_labels)
                except:
                    add_to_loss2 = 0
                # when all ranks are in order, total_correct == total_labels & 1 - 1 = 0; nothing gets added to loss
            else:
                add_to_loss1, add_to_loss2 = 0, 0
        else:
            #############################################################
            # 1 - apply sigmoid to logits to get probabilities
            temp = torch.nn.Sigmoid()(logits).cpu()
            ranks = ranks.cpu()
            n_labels = torch.max(ranks, axis=1)[0]  # shape: batch_size -- when not using doc batching

            # 2 - separate true probs from remaining probs; rank true probs (there are no rankings for negative labels)
            true_label_probs = [torch.flip(temp[r, m], dims=(0,)) for r, m in
                                enumerate([torch.argsort(ranks)[i, -j:] for i, j in
                                           enumerate(n_labels) ])]  # - when not using doc batching
            # ^ probabilities of all of the true positive labels, sorted in descending rank order

            remaining_probs = [temp[r, m] for r, m in
                               enumerate([torch.argsort(ranks)[i, :-j] if j>0 else torch.argsort(ranks)[i, :]  for i, j in
                                          enumerate(n_labels)])]  # - when not using doc batching
            # ^ probabilities of all of the true negative labels

            # 3 - check if lowest-ranking true positive probability is higher than all negative class probs
            lowest_ranking_probs = torch.Tensor([t[-1] for t in true_label_probs])  # - when not using doc batching

            highest_remaining_probs = torch.Tensor(
                [torch.max(t) for t in remaining_probs])  # - when not using doc batching

            temp = lowest_ranking_probs - highest_remaining_probs  # - when not using doc batching
            temp[temp < 0] = 0
            add_to_loss1 = torch.sum(temp).item()

            # 4 - check that the probabilities are ranked in order
            vals = self.check_rankings(true_label_probs)  # - when not using doc batching

            total_correct = sum(sum(t) for t in vals).item()
            total_labels = sum(len(t) for t in vals)
            add_to_loss2 = 1 - (total_correct / total_labels)
            # when all ranks are in order, total_correct == total_labels & 1 - 1 = 0; nothing gets added to loss
        return add_to_loss1 + add_to_loss2

    def check_rankings(self, ranking_probs):
        to_return = []
        for probs in ranking_probs:
            while len(probs) > 1:
                to_return.append(probs[0] > torch.Tensor([probs[i] for i in range(1, len(probs))]))
                probs = probs[1:]
        return to_return


class BalancedBCEWithLogitsLoss(nn.Module):
    
    def __init__(self, grad_clip=False, weights=None, reduction='mean'):
        super(BalancedBCEWithLogitsLoss, self).__init__()
        self.grad_clip = grad_clip
        self.weights = weights
        self.reduction = reduction
    
    def forward(self, logits, labels):
        # logits: shape(batch_size, num_classes), dtype=float
        # labels: shape(batch_size, num_classes), dtype=float
        # labels must be a binary valued tensor
        assert logits.shape == labels.shape, "logits shape %r != labels shape %r" % (logits.shape, labels.shape)

        if self.weights is not None:
            # then we have class weights to make use of
            s_logits = torch.nn.Sigmoid()(logits)
            log_s_logits = torch.log(s_logits)
            weighted_targets = self.weights * labels


            one_minus_targets = 1 - labels
            log_one_minus_s_logits = torch.log(1-s_logits)

            # weighted_loss = torch.matmul(weighted_targets, log_s_logits) + torch.matmul(one_minus_targets, log_one_minus_s_logits)
            weighted_loss = -((weighted_targets * log_s_logits) + (one_minus_targets * log_one_minus_s_logits))

            assert weighted_loss.shape == logits.shape
            weighted_loss = weighted_loss.mean()

        else:
            weighted_loss = False

        # number of classes
        nc = labels.shape[1]

        # number of positive classes per example in batch
        npos_per_example = labels.sum(1)                # shape: [batch_size]
        
        # alpha: ratio of negative classes per example in batch
        alpha = (nc - npos_per_example) / npos_per_example
        alpha[alpha == float("Inf")] = 0
        alpha = alpha.unsqueeze(1).expand_as(labels)    # shape: [batch_size, num_classes]
        
        # positive weights
        pos_weight = labels * alpha
        
        # to avoid gradients vanishing
        if self.grad_clip:
            logits = logits.clamp(min=-14.0, max=14.0)
        
        proba = torch.sigmoid(logits)
        # see https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss for loss eq.
        loss = -(torch.log(proba) * pos_weight + torch.log(1. - proba) * (1. - labels))
        # the labels which are supposed to be positive get more weight added to them
        loss = loss.mean()
        if weighted_loss and self.reduction == 'mean':
            loss = (loss + weighted_loss)/2
        elif weighted_loss and self.reduction == 'sum':
            loss += weighted_loss

        
        return loss
