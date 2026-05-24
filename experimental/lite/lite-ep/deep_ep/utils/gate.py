import torch


def generate_topk_idx(rank_count: torch.Tensor, num_tokens: int, num_experts: int, num_ranks: int, num_topk: int) -> torch.Tensor:
    """
    Map rank count to expert indices
    """
    assert torch.equal(torch.sum(rank_count, dim=1), torch.ones(num_tokens, dtype=torch.int, device='cuda') * num_topk)
    assert (num_tokens, num_ranks) == rank_count.shape
    num_experts_per_rank = num_experts // num_ranks

    # Generate base value
    base_vals = torch.arange(num_experts, device='cuda').view(1, num_ranks, num_experts_per_rank).expand(num_tokens, num_ranks, num_experts_per_rank)

    # Randomize the ordering within each row
    rand_vals = torch.rand(num_tokens, num_ranks, num_experts_per_rank, device='cuda')
    perm_indices = torch.argsort(rand_vals, dim=-1)
    permuted = torch.gather(base_vals, 2, perm_indices)

    # Create the mask
    k_idx = torch.arange(num_experts_per_rank, device='cuda').view(1, 1, num_experts_per_rank).expand(num_tokens, num_ranks, num_experts_per_rank)
    rank_count_expanded = rank_count.unsqueeze(2).expand(num_tokens, num_ranks, num_experts_per_rank)
    mask = k_idx < rank_count_expanded

    # Get the final indices by masking and reshaping
    selected = permuted[mask]  # (num_tokens * num_topk,)
    topk_idx = selected.view(num_tokens, num_topk)

    return topk_idx


def generate_rank_count(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float) -> torch.Tensor:
    """
    Generate rank count tensor for a given number of tokens, experts, ranks, and top-k.

    This function generates a tensor of shape `(num_tokens, num_ranks)` where each element `[i, j]` represents
    the number of topk experts that token `i` have on rank `j`. The distribution is such that
    one special rank gets `ratio` times more traffic than the others.
    """
    num_experts_per_rank = num_experts // num_ranks
    num_normal_ranks = num_ranks - 1

    assert ratio >= 1.0, 'ratio must be no less than 1.0'

    # Generate rank count of each token from random distribution
    random_scores = torch.rand(num_tokens, num_experts, device='cuda')
    topk_weights_, topk_indices = torch.topk(random_scores, num_topk, dim=1, largest=True, sorted=False)
    topk_indices //= num_experts_per_rank
    sorted_topk_indices = torch.sort(topk_indices, dim=1)[0]
    topk_indices_diff_mask = sorted_topk_indices[:, 1:] != sorted_topk_indices[:, :-1]
    a = topk_indices_diff_mask.sum(dim=1) + 1

    # Upper bound for this generating algorithm
    upper_bound_per_token = int(num_normal_ranks / ratio) + 1

    # Clamp the value in range [1, upper_bound_per_token] for each token
    a = torch.clamp(a, None, upper_bound_per_token)

    # Consider the special rank
    sum_a = torch.sum(a).item()
    normal_token_count = int(sum_a / (num_normal_ranks + ratio))
    special_token_count = sum_a - normal_token_count * num_normal_ranks
    special_token_count = min(special_token_count, int(normal_token_count * ratio) + 1)

    # Tokens that the special rank must be in topk
    must_mask = (a == num_ranks)
    must_count = int(must_mask.sum().item())
    special_token_count = max(must_count, special_token_count)
    assert must_count <= special_token_count, 'Too many tokens with full rank assignment'

    # Tokens that the special rank can optionally be in topk
    optional_token_indices = torch.where(must_mask == 0)[0]
    optional_token_indices = optional_token_indices[torch.randperm(num_tokens - must_count, device='cuda')][:special_token_count - must_count]
    must_token_indices = torch.where(must_mask != 0)[0]
    special_token_row_index = torch.cat(([must_token_indices, optional_token_indices]))

    # Generate permutations for normal ranks
    rank_perm = (torch.randperm(num_normal_ranks, device='cuda') + 1).repeat(num_tokens * num_topk // num_normal_ranks + 1)

    # Compute cumulative sum of a to get starting indices in b for each row
    a_cumsum = torch.cumsum(torch.cat((torch.tensor([0], device='cuda'), a)), dim=0)
    row_starts = a_cumsum[:-1]  # Starting indices for each row in b, shape (n,)

    # Insert special rank index into the permutation for special tokens
    rank_perm_with_special_rank = torch.zeros(num_tokens * num_topk, dtype=torch.long, device='cuda')  # (n * k,)
    special_token_mask = torch.zeros(num_tokens * num_topk, dtype=torch.bool, device='cuda')
    special_token_flattened_row_index = row_starts[special_token_row_index]
    special_token_mask[special_token_flattened_row_index] = 1
    all_indices = torch.arange(num_tokens * num_topk, device='cuda')
    non_special_indices = all_indices[special_token_mask != True]
    rank_perm_with_special_rank[non_special_indices] = rank_perm[:len(non_special_indices)]

    # Create column index grids
    col_idx = torch.arange(num_topk, device='cuda').view(1, num_topk)  # (1, num_topk)

    # Compute modulo indices: col_idx % a[i] for each row
    # torch.max is used to avoid zeros in case a[i] = 0 (which happens when the only topk rank is the special rank)
    mod_idx = col_idx % a.view(num_tokens, 1)  # (n, num_topk)

    # Compute indices in b: row_start + (col % a[i])
    b_indices = row_starts.view(num_tokens, 1) + mod_idx  # (n, k)

    # Gather values from b using computed indices
    result = rank_perm_with_special_rank[b_indices]

    # Shuffle rows randomly to avoid any pattern
    shuffle_indices = torch.randperm(num_tokens, device='cuda')
    result = result[shuffle_indices]  # Shuffle rows

    # Create rank count tensor
    rank_count = torch.zeros((num_tokens, num_ranks), dtype=torch.int32, device='cuda')
    rank_count.scatter_add_(dim=1, index=result, src=torch.ones_like(result, dtype=torch.int32))
    return rank_count


def get_precise_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float):
    """
    Generate precise unbalanced scores for testing.

    Note that this function generates a distribution with precise unbalanced distribution,
    which **differs from real distribution**.
    """
    # Generate num topk experts for each rank
    rank_count = generate_rank_count(num_tokens, num_experts, num_ranks, num_topk, ratio)

    # Generate scores in a low distribution
    threshold = 0.9
    scores = torch.empty((num_tokens, num_experts), dtype=torch.float32, device='cuda')
    scores.uniform_(to=threshold)

    # Generate topk indices and change their scores to a high distribution
    topk_idx = generate_topk_idx(rank_count, num_tokens, num_experts, num_ranks, num_topk)
    topk_scores = torch.empty((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    topk_scores.uniform_(threshold + 1e-6, 1.0)
    row_idx = torch.arange(num_tokens).unsqueeze(1).expand(num_tokens, num_topk)
    scores[row_idx, topk_idx] = topk_scores
    return scores


def get_scores_by_factor(num_tokens: int, num_experts: int, num_ranks: int, factor: float) -> torch.Tensor:
    num_experts_per_rank = num_experts // num_ranks
    scores = torch.empty((num_tokens, num_experts), dtype=torch.float32, device='cuda')
    scores[:, :num_experts_per_rank].uniform_(to=factor)
    scores[:, num_experts_per_rank:].uniform_(to=1)
    return scores


def map_unbalanced_ratio_to_factor(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float) -> float:
    num_iterations = 20
    factor_l, factor_r = 1.0, 100.0

    num_experts_per_rank = num_experts // num_ranks
    for _i in range(num_iterations):
        factor_mid = (factor_l + factor_r) / 2
        scores = get_scores_by_factor(num_tokens, num_experts, num_ranks, factor_mid)
        _, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        rank_idx = topk_idx // num_experts_per_rank
        one_hot = torch.nn.functional.one_hot(rank_idx, num_ranks)
        counts = one_hot.any(dim=1).to(torch.float).sum(dim=0)
        if counts[0].item() > counts[1:].mean().item() * ratio:
            factor_r = factor_mid
        else:
            factor_l = factor_mid
    return factor_l


def get_random_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float):
    """Generate unbalanced scores with a given ratio.
    """
    factor = 1.0
    if ratio != 1.0:
        factor = map_unbalanced_ratio_to_factor(num_tokens, num_experts, num_ranks, num_topk, ratio)
    return get_scores_by_factor(num_tokens, num_experts, num_ranks, factor)


def get_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float, precise: bool):
    if precise:
        return get_precise_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio)
    else:
        return get_random_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio)
