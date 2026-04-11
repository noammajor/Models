import torch
def apply_mask(input_data, masks):
    """
    Handles both Teacher (Dict) and Student (Tensor) masking.
    
    :param input_data: Either a Dict containing 'data_patches' or a raw Tensor [B*F, P, D]
    :param masks: The indices to keep [B, M]
    """
    is_dict = isinstance(input_data, dict)
    x = input_data["data_patches"] if is_dict else input_data
    masks = masks.to(x.device)
    B_total, P, D = x.shape
    B_orig = input_data.get("orig_B", masks.size(0)) if is_dict else masks.size(0)
    F = B_total // B_orig
    if F > 1:
        current_masks = masks.unsqueeze(1).repeat(1, F, 1).view(B_total, -1)
    else:
        current_masks = masks
    mask_idx = current_masks.unsqueeze(-1).expand(-1, -1, D)
    masked_x = torch.gather(x, dim=1, index=mask_idx)
    if is_dict:
        input_data["data_patches"] = masked_x
        return input_data
    else:
        return masked_x