from torch.utils.data import DataLoader
import pandas as pd
import gzip
import numpy as np
from data_loaders.data_puller import DataPuller

import torch


def get_jepa_loaders(
    data_paths,
    patch_size,
    batch_size,
    ratio_patches,
    mask_ratio,
    masking_type,
    num_semantic_tokens,
    input_variables,
    timestamp_cols,
    type_data,
    val_prec=0.1,
    test_prec=0.25
    ):
    dataloader = DataPuller(data_paths=data_paths,
                            patch_size=patch_size,
                            batch_size=batch_size,
                            ratio_patches=ratio_patches,
                            mask_ratio=mask_ratio,
                            masking_type=masking_type,
                            num_semantic_tokens=num_semantic_tokens,
                            input_variables=input_variables,
                            timestamp_cols=timestamp_cols,
                            type_data=type_data,
                            val_prec=val_prec,
                            test_prec=test_prec
                            )

    dataloader = DataLoader(dataloader,
                            batch_size=batch_size,
                            shuffle=True)

    return dataloader