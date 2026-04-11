"""
Next-Token (Patch) Prediction pretraining using causal PatchTST.

Uses the same JEPA-style data loaders (MonashDataPullerJEPA / DataPullerDJepa)
so patches are already created — no patching logic needed here.

Data loader output:  (patches, context_idx, target_idx)
  patches: [B x P x P_L x F]   (P=num_patches, P_L=patch_len, F=features/vars)

PatchTST input:      [B x P x F x P_L]  (permute last two dims)

Pretraining objective:
    With causal attention, the representation at position i sees only patches
    0..i.  We predict patch i+1 from position i.
    Loss = MSE( pred[:, :-1, :, :], patches[:, 1:, :, :] )
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn

# ── path setup ────────────────────────────────────────────────────────────────
_NPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_NPT_DIR)
_DATALOADER_DIR  = os.path.join(_ROOT_DIR, 'Utils')

for _p in [_NPT_DIR, _ROOT_DIR, _DATALOADER_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.patchTST import PatchTST
from data_loaders.data_puller import MonashDataPullerJEPA, DataPullerDJepa, SyntheticArrowDataPullerJEPA


# ── normalization ─────────────────────────────────────────────────────────────

def _instance_norm(x, eps=1e-6):
    """
    Per-instance normalization over the patch and patch-length dims.
    x: [B x P x P_L x F]
    returns: normed x, mean, std
    """
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2),  keepdim=True) + eps
    return (x - mean) / std, mean, std


# ── model ─────────────────────────────────────────────────────────────────────

def get_model(config, c_in, device):
    horizon_patches  = config['horizon_t']
    context_patches  = config['ratio_patches'] - horizon_patches
    patch_len        = config['patch_size']
    model = PatchTST(
        c_in=c_in,
        target_dim=patch_len,       # not used by NTPHead, kept for signature compat
        patch_len=patch_len,
        stride=patch_len,           # non-overlapping (same as JEPA data loader)
        num_patch=context_patches,  # encoder sees only context patches
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_model=config['d_model'],
        shared_embedding=True,
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        head_dropout=config['head_dropout'],
        act=config['act'],
        head_type='ntp',
        causal=True,
        res_attention=False,
        horizon_patches=horizon_patches,  # head outputs exactly horizon patches
    ).to(device)
    print(f'model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    return model


# ── save path ────────────────────────────────────────────────────────────────

def _model_fname(config, dset):
    return (
        f"ntp_pretrained"
        f"_patch{config['patch_size']}"
        f"_patches{config['ratio_patches']}"
        f"_epochs{config['num_epochs']}"
        f"_model{config.get('pretrained_model_id', 1)}"
    )


# ── checkpoint helpers ───────────────────────────────────────────────────────

def save_model(model, optimizer, epoch, path_save):
    checkpoint_dir = os.path.dirname(path_save)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Created directory: {checkpoint_dir}")
        except Exception as e:
            print(f"Could not create directory {checkpoint_dir}: {e}")
            return

    save_dict = {
        "epoch":   epoch,
        "encoder": model.backbone.state_dict(),
    }
    try:
        print(f"Saving checkpoint to: {path_save}")
        torch.save(save_dict, path_save)
        print(f"Checkpoint saved: {path_save}")
    except Exception as e:
        print(f"Problem saving checkpoint: {e}")


# ── training ─────────────────────────────────────────────────────────────────

def pretrain_ntp(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    pretrain_dset = config.get('pretrain_source') or config.get('pretrain_dataset', 'monash')

    # ── data ─────────────────────────────────────────────────────────────────
    if pretrain_dset in ('monash', 'synthetic', 'monash+synthetic'):
        _cfg = dict(config)
        if pretrain_dset in ('monash', 'monash+synthetic'):
            monash_dir = config['monash_data_dir']
            if not os.path.isabs(monash_dir):
                monash_dir = os.path.normpath(os.path.join(_NPT_DIR, monash_dir))
            _cfg['monash_data_dir'] = monash_dir
        if pretrain_dset in ('synthetic', 'monash+synthetic'):
            synth_dir = config['synthetic_data_dir']
            if not os.path.isabs(synth_dir):
                synth_dir = os.path.normpath(os.path.join(_NPT_DIR, synth_dir))
            _cfg['synthetic_data_dir'] = synth_dir

        if pretrain_dset in ('monash', 'monash+synthetic'):
            train_dataset = MonashDataPullerJEPA(_cfg, which='train')
            val_dataset   = MonashDataPullerJEPA(_cfg, which='val')
        if pretrain_dset in ('synthetic', 'monash+synthetic'):
            import torch.utils.data as _tud
            syn_train = SyntheticArrowDataPullerJEPA(_cfg, which='train')
            syn_val   = SyntheticArrowDataPullerJEPA(_cfg, which='val')
            if pretrain_dset == 'monash+synthetic':
                train_dataset = _tud.ConcatDataset([train_dataset, syn_train])
                val_dataset   = _tud.ConcatDataset([val_dataset,   syn_val])
            else:
                train_dataset, val_dataset = syn_train, syn_val
        c_in = 1
    else:
        from dataset_registry import get_dataset_info
        sys.path.insert(0, _ROOT_DIR)
        ds = get_dataset_info(pretrain_dset)
        n_groups = len(ds['jepa_groups'])
        _cfg = dict(config,
                    path_data         = [ds['csv_path']] * n_groups,
                    timestampcols     = [ds['timestamp_col']] * n_groups,
                    input_variables   = ds['jepa_groups'],
                    stride            = config.get('stride', config['patch_size']))
        train_dataset = DataPullerDJepa(**{k: _cfg[k] for k in [
            'path_data', 'patch_size', 'batch_size', 'ratio_patches',
            'mask_ratio', 'masking_type', 'input_variables', 'timestampcols',
        ]}, type_data='train',
            val_prec=config.get('val_prec', 0.1),
            test_prec=config.get('test_prec', 0.1),
            stride=_cfg['stride'],
            num_blocks=config.get('num_blocks', 1))
        val_dataset = copy.copy(train_dataset); val_dataset.which = 'val'
        c_in = len(train_dataset[0][0][0])  # F from [P, P_L, F]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config.get('num_workers', 0))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,   batch_size=config['batch_size'], shuffle=False,
        num_workers=config.get('num_workers', 0))

    print(f'train batches: {len(train_loader)}  val batches: {len(val_loader)}')

    # ── model + optimizer ─────────────────────────────────────────────────────
    model = get_model(config, c_in, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=config['lr'] * 0.1)
    loss_fn = nn.MSELoss()

    save_dir  = os.path.join(_NPT_DIR, 'saved_models', pretrain_dset, 'ntp', f"layers{config['n_layers']}")
    os.makedirs(save_dir, exist_ok=True)
    save_name = _model_fname(config, pretrain_dset)

    best_val_loss = float('inf')
    records = []

    context_patches = config['ratio_patches'] - config['horizon_t']  # 27 - 6 = 21

    for epoch in range(config['num_epochs']):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for patches, _, _ in train_loader:
            patches = patches.float().to(device)   # [B x P x P_L x F]

            if config.get('revin', True):
                patches, _, _ = _instance_norm(patches)

            # PatchTST expects [B x P x F x P_L]
            x = patches.permute(0, 1, 3, 2)

            # Feed context, predict horizon — pred is exactly [B x horizon_patches x F x P_L]
            pred = model(x[:, :context_patches])
            loss = loss_fn(pred, x[:, context_patches:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for patches, _, _ in val_loader:
                patches = patches.float().to(device)
                if config.get('revin', True):
                    patches, _, _ = _instance_norm(patches)
                x    = patches.permute(0, 1, 3, 2)
                pred = model(x[:, :context_patches])
                val_losses.append(loss_fn(pred, x[:, context_patches:]).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        scheduler.step()
        records.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f'epoch {epoch+1}/{config["num_epochs"]}  '
              f'train={train_loss:.6f}  val={val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch + 1,
                       os.path.join(save_dir, save_name + '.pt'))
            print(f'  → saved (best val={best_val_loss:.6f})')
        save_model(model, optimizer, epoch + 1,
                   os.path.join(save_dir, save_name + f'_epoch{epoch+1}.pt'))
    pd.DataFrame(records).to_csv(
        os.path.join(save_dir, save_name + '_losses.csv'),
        index=False, float_format='%.6f')
    print(f'\nDone. Best val loss: {best_val_loss:.6f}')
    print(f'Model: {os.path.join(save_dir, save_name)}.pt')
    return os.path.join(save_dir, save_name + '.pt')


if __name__ == '__main__':
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        'config_ntp', os.path.join(_NPT_DIR, 'config_ntp.py'))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    pretrain_ntp(dict(_mod.config))
