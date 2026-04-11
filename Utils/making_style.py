import torch

def get_mask_style(B, num_patches, type="bernoulli", p=0.5, device='cpu', num_blocks=3):
    """
    B: Batch size
    num_patches: Total number of patches
    p: fraction of patches to mask per block (= target ratio per block)
    num_blocks: number of blocks for "multi_block" masking

    Masking types:
      "bernoulli"   — random uniform masking
      "block"       — random contiguous block masking
      "causal"      — forecasting masking: context = first (1-p)*N patches,
                      target = last p*N patches. Same split for every sample
                      in the batch (deterministic given num_patches and p).
      "multi_block" — num_blocks non-overlapping random blocks each of size
                      int(num_patches * p). Total masked = num_blocks * block_size.
    """
    if type == "bernoulli":
        return get_bernoulli_indices(B, num_patches, p, device)
    elif type == "block":
        return get_block_indices(B, num_patches, int(num_patches * p), device)
    elif type == "causal":
        return get_causal_indices(B, num_patches, int(num_patches * p), device)
    elif type == "multi_block":
        return get_multi_block_indices(B, num_patches, num_blocks, int(num_patches * p), device)

def get_bernoulli_indices(B, num_patches, p, device):
    num_keep = int(num_patches * (1 - p))

    # Generate random indices for the whole batch
    # Each row is a shuffled version of [0, ..., num_patches-1]
    batch_indices = torch.stack([torch.randperm(num_patches, device=device) for _ in range(B)])

    context_idx = batch_indices[:, :num_keep]
    target_idx = batch_indices[:, num_keep:]

    return context_idx, target_idx

def get_block_indices(B, num_patches, block_size, device):
    context_list = []
    target_list = []

    for _ in range(B):
        all_idx = torch.arange(num_patches, device=device)
        start = torch.randint(0, num_patches - block_size+1, (1,)).item()

        t_idx = all_idx[start : start + block_size]
        c_idx = torch.cat([all_idx[:start], all_idx[start + block_size :]])

        context_list.append(c_idx)
        target_list.append(t_idx)

    return torch.stack(context_list), torch.stack(target_list)


def get_causal_indices(B, num_patches, horizon, device):
    """Forecasting masking: context = first (num_patches - horizon) patches,
    target = last horizon patches. Every sample sees the same positions (the
    split is deterministic) but different patch values, like forecasting."""
    num_ctx = num_patches - horizon
    context_list = []
    target_list = []

    for _ in range(B):
        all_idx = torch.arange(num_patches, device=device)
        context_list.append(all_idx[:num_ctx])
        target_list.append(all_idx[num_ctx:])

    return torch.stack(context_list), torch.stack(target_list)


def get_multi_block_indices(B, num_patches, num_blocks, block_size, device):
    """Place num_blocks non-overlapping blocks of block_size in [0, num_patches).

    Uses a bijection trick: sample num_blocks sorted values from [0, gap_space]
    then offset each by i*block_size to guarantee non-overlap. This gives a
    uniform distribution over all valid placements with no rejection sampling.

    gap_space = num_patches - num_blocks * block_size  (free positions)
    start_i   = u_i + i * block_size  (guaranteed: start_{i+1} >= start_i + block_size)

    context: all patches not covered by any block
    target:  union of all blocks (exactly num_blocks * block_size patches)
    """
    # Clamp block_size so all blocks fit: num_blocks * block_size <= num_patches
    block_size = min(block_size, num_patches // num_blocks)
    gap_space = num_patches - num_blocks * block_size
    context_list = []
    target_list = []

    for _ in range(B):
        u = torch.randint(0, gap_space + 1, (num_blocks,)).sort().values.tolist()
        starts = [u[i] + i * block_size for i in range(num_blocks)]

        # Build sorted target indices
        target_set = set()
        for s in starts:
            target_set.update(range(s, s + block_size))

        target_idx = torch.tensor(sorted(target_set), device=device)
        context_idx = torch.tensor([i for i in range(num_patches) if i not in target_set], device=device)

        context_list.append(context_idx)
        target_list.append(target_idx)

    return torch.stack(context_list), torch.stack(target_list)
