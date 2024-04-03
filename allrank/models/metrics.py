import numpy as np
import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE


def ndcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE,
         filler_value=1.0):
    """
    Normalized Discounted Cumulative Gain at k.

    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param filler_value: a filler NDCG value to use when there are no relevant items in listing
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = filler_value  # if idcg == 0 , set ndcg to filler_value

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def mrr(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE):
    """
    Mean Reciprocal Rank at k.

    Compute MRR at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: MRR values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    values, indices = torch.max(true_sorted_by_preds, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t().expand(len(y_true), len(ats))

    ats_rep = torch.tensor(data=ats, device=indices.device, dtype=torch.float32).expand(len(y_true), len(ats))

    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0

    result = result * within_at_mask

    return result

def recall(y_pred, y_true, ats=None, min_relevance=1, padding_indicator=PADDED_Y_VALUE):
    """
    Recall at k.

    Compute Recall at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param min_relevance: minimum relevance value to be considered as relevant
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: Recall values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    recalls = []
    for i in range(len(ats)):

        y_true = y_true.clone()
        y_pred = y_pred.clone()

        if ats is None:
            ats = [y_true.shape[1]]

        ats = np.array(ats)
        ats = np.where(ats == 0, y_true.shape[1], ats)

        true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

        ats_rep = torch.tensor(data=ats[i], device=true_sorted_by_preds.device, dtype=torch.float32).expand(
            y_true.shape)
        indices = torch.arange(0, y_true.shape[1], device=true_sorted_by_preds.device, dtype=torch.float32).expand(
            y_true.shape)
        within_at_mask = (indices < ats_rep).type(torch.float32)

        masked_true_sorted_by_preds = true_sorted_by_preds * within_at_mask
        relevant_retrieved = masked_true_sorted_by_preds >= min_relevance
        relevant_total = (true_sorted_by_preds >= min_relevance).type(torch.float32).sum(dim=1, keepdim=True)
        recalls.append(
            torch.sum(relevant_retrieved, dim=1, keepdim=True) / relevant_total)
        zero_mask = relevant_total == 0
        recalls[i][zero_mask] = 0
    return torch.cat(tuple(recalls), 1)

def map(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE, cutoff=1):
    """
    Map at k.

    Compute mean average precision at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param cutoff: the minimum relevance value to be considered as relevant
    :return: Mean average precision values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    # TODO: Add support for multiple ats values in one list
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    max_at = min(max(ats), y_true.shape[1])

    # sort y_true using y_preds
    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    # return shape [n_batches, n_ats]
    res = np.zeros((len(y_pred), len(ats)))

    for i, r in enumerate(true_sorted_by_preds):
        # count current sum of precisions
        running_sum = 0
        max_len = len(r) if 0 in ats else max_at
        running_sums = np.zeros((max_len))
        # count current amount of true positives
        current_relevant = 0
        for at in range(max_len):
            if r[at] >= cutoff:
                current_relevant += 1
                # add current precision
                running_sum += current_relevant / (at + 1)
            if current_relevant > 0:
                running_sums[at] = running_sum / current_relevant
        res[i] = running_sums[np.array(ats) - 1]

    return torch.tensor(res)

def precision(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE, cutoff=1, no_torch=True):
    """
    Recall at k.

    Compute Precision at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: Precision values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    # TODO: Add support for multiple ats values in one list
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    ats = np.array(ats)
    ats = np.where(ats == 0, y_true.shape[1], ats)

    if no_torch:
        max_at = min(max(ats), y_true.shape[1])

        # sort y_true using y_preds
        true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

        # return shape [n_batches, n_ats]
        res = np.zeros((len(y_pred), len(ats)))

        for i, r in enumerate(true_sorted_by_preds):
            running_sums = np.zeros((max_at))
            # count current amount of true positives
            current_correct = 0
            for at in range(max_at):
                if r[at] >= cutoff:
                    current_correct += 1
                # add current precision
                running_sums[at] = current_correct / (at + 1)
            res[i] = running_sums[np.array(ats) - 1]

        return torch.tensor(res)

    else:

        true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator) \
            .unsqueeze(dim=0).expand(len(ats), y_true.shape[0], y_true.shape[1])

        print(true_sorted_by_preds, true_sorted_by_preds.shape)

        values, indices = torch.max(true_sorted_by_preds, dim=1)
        indices = torch.arange(0, y_true.shape[1], device=true_sorted_by_preds.device, dtype=torch.float32) \
            .unsqueeze(dim=0).expand(y_true.shape[0], y_true.shape[1]) \
            .unsqueeze(dim=0).expand(len(ats), y_true.shape[0], y_true.shape[1])
        print(indices)

        ats_rep = torch.tensor(data=ats, device=indices.device, dtype=torch.float32).expand(len(ats), y_true.shape[1]) \
            .unsqueeze(dim=1).expand(len(ats), y_true.shape[0], y_true.shape[1])

        within_at_mask = (indices < ats_rep).type(torch.float32)

        true_positives = (true_sorted_by_preds >= cutoff).type(torch.float32) * within_at_mask

        result = torch.sum(true_positives, dim=1) / torch.sum(within_at_mask, dim=1)

        zero_sum_mask = torch.sum(values) == 0.0
        result[zero_sum_mask] = 0.0

        return result
def rmse(y_pred, y_true, ats=None, no_of_levels=1, padded_value_indicator=PADDED_Y_VALUE):
    """
    Pointwise RMSE loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param no_of_levels: number of unique ground truth values
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # This was already present and tested in losses.py    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    valid_mask = (y_true != padded_value_indicator).type(torch.float32)

    y_true[mask] = 0
    y_pred[mask] = 0

    errors = (y_true - no_of_levels * y_pred)

    squared_errors = errors ** 2

    mean_squared_errors = torch.sum(squared_errors, dim=1) / torch.sum(valid_mask, dim=1)

    rmses = torch.sqrt(mean_squared_errors)

    rmses = rmses.unsqueeze(dim=0).t()

    return rmses.detach()