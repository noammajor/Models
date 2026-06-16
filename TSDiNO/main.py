import copy
import numpy as np
from config import config as cfg
import pandas as pd
import os
import torch
from torch import nn
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.patchTST import PatchTST, PatchReconDecoder
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import time
import datetime
import math
import sys
from pathlib import Path
import json
import data_agumentation as aug
from models.layers.Dino_Head import DINOHead
import utils.util as utils
import dataPuller as dpuller
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset


def train_TS_DINO(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    #print("git:\n  {}\n".format(utils.get_sha()))
    #print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    #cudnn.benchmark = True

    #-------------DATA-----------------
    dataAugmentationDino = DataAugmentationDino(
        global_crops = cfg['global_crops'],
        local_crops  = cfg['local_crops'],
        dwt_cfg      = cfg,
    )

    # ── Select pretraining dataset ─────────────────────────────────────────────
    _pretrain_source = cfg.get('pretrain_source') or (
        'monash' if cfg.get('pretrain_on_monash', False) else None
    )
    _shared_kwargs = dict(
        split     = 'train',
        transform = dataAugmentationDino,
        batch_size= args.num_patches,
        patch_size= args.patch_len,
        step_size = args.step_size,
        min_len   = cfg.get('monash_min_len', 512),
    )
    _synth_kwargs = dict(_shared_kwargs, window_step=cfg.get('window_step', None))
    if _pretrain_source in ('monash', 'monash+synthetic'):
        print("Using Monash dataset for DINO pretraining")
        combined_dataset = dpuller.MonashDataPuller(
            data_dir = cfg['monash_data_dir'], **_shared_kwargs)
        if _pretrain_source == 'monash+synthetic':
            print("Using Monash + Synthetic (mix) datasets for DINO pretraining")
            _mix_dir = cfg.get('synthetic_mix_data_dir', cfg['synthetic_data_dir'])
            syn_dataset = dpuller.SyntheticArrowDataPuller(
                data_dir = _mix_dir, **_synth_kwargs)
            combined_dataset = ConcatDataset([combined_dataset, syn_dataset])
    elif _pretrain_source == 'synthetic':
        print("Using Synthetic dataset for DINO pretraining")
        combined_dataset = dpuller.SyntheticArrowDataPuller(
            data_dir = cfg['synthetic_data_dir'], **_synth_kwargs)
    if _pretrain_source is not None:
        print(f"Pretrain dataset: {len(combined_dataset)} windows")
    elif 'UCI HAR' in args.data_path:
        print("Using UCI HAR Dataset for DINO training")
        combined_dataset = dpuller.DataPullerUCIDINO(
            data_dir=args.data_path,
            split='train',
            transform=dataAugmentationDino,
            batch_size=args.num_patches,
            patch_size=args.patch_len,
            step_size=args.step_size,
            c_in=args.c_in
        )
    else:
        print("Using CSV datasets for DINO training")
        _shared_dir = str(Path(__file__).parent.parent / "shared")
        if _shared_dir not in sys.path:
            sys.path.insert(0, _shared_dir)
        from data_loaders.data_puller import PatchTSTPretrainAdapter
        _seq_len = args.num_patches * args.patch_len
        dataset1 = PatchTSTPretrainAdapter(
            csv_path=args.data_path,
            split='train',
            seq_len=_seq_len,
            patch_size=args.patch_len,
            transform=dataAugmentationDino,
        )
        dataset2 = PatchTSTPretrainAdapter(
            csv_path=args.data_path_forecast_training,
            split='train',
            seq_len=_seq_len,
            patch_size=args.patch_len,
            transform=dataAugmentationDino,
        )
        combined_dataset = ConcatDataset([dataset1, dataset2])

    # ── val dataset (same source, split='val') ────────────────────────────────
    _val_kwargs = dict(_shared_kwargs, split='val')
    if _pretrain_source in ('monash', 'monash+synthetic'):
        val_dataset = dpuller.MonashDataPuller(data_dir=cfg['monash_data_dir'], **_val_kwargs)
        if _pretrain_source == 'monash+synthetic':
            _val_synth_kwargs = dict(_val_kwargs, window_step=cfg.get('window_step', None))
            syn_val = dpuller.SyntheticArrowDataPuller(data_dir=cfg['synthetic_data_dir'], **_val_synth_kwargs)
            val_dataset = ConcatDataset([val_dataset, syn_val])
    elif _pretrain_source == 'synthetic':
        _val_synth_kwargs = dict(_val_kwargs, window_step=cfg.get('window_step', None))
        val_dataset = dpuller.SyntheticArrowDataPuller(data_dir=cfg['synthetic_data_dir'], **_val_synth_kwargs)
    elif 'UCI HAR' in args.data_path:
        val_dataset = None  # UCI HAR — no val split
    else:
        # CSV in-domain: use val split so checkpoint_best.pth gets saved
        val_dataset = PatchTSTPretrainAdapter(
            csv_path  = args.data_path,
            split     = 'val',
            seq_len   = args.num_patches * args.patch_len,
            patch_size= args.patch_len,
            transform = dataAugmentationDino,
        )

    _is_distributed = utils.is_dist_avail_and_initialized()
    data_loader = torch.utils.data.DataLoader(
        combined_dataset,
        sampler=torch.utils.data.distributed.DistributedSampler(combined_dataset, shuffle=True) if _is_distributed else torch.utils.data.RandomSampler(combined_dataset),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    ) if val_dataset is not None and len(val_dataset) > 0 else None



    #------------- Student - Teacher network ---------------

    student = PatchTST(
        c_in= args.c_in,
        target_dim=args.pred_len,
        patch_len=args.patch_len,
        num_patch=args.num_patches,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.embed_dim,
        shared_embedding=True,
        d_ff=args.d_ff,                        
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='gelu',
        head_type='Dino',
        res_attention=False,
        drop_path_rate=args.drop_path_rate,
        step_size=args.step_size
        )
    teacher = PatchTST(
        c_in= args.c_in,
        target_dim=args.pred_len,
        patch_len=args.patch_len,
        num_patch=args.num_patches,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.embed_dim,
        shared_embedding=True,
        d_ff=args.d_ff,                        
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='gelu',
        head_type='Dino',
        res_attention=False,
        drop_path_rate=0.0,
        step_size=args.step_size)
    embed_dim = student.backbone.d_model
    student = utils.TSMultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.TSMultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head, norm_last_layer=args.norm_last_layer),
    )
    # move networks to gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    student, teacher = student.to(device), teacher.to(device)
    # synchronize batch norms (if any)
    device_ids = [args.gpu] if torch.cuda.is_available() else None
    if utils.has_batchnorms(student):
        # Around line 190 in train_TS_DINO
        if torch.cuda.is_available():
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        else:
            print("Skipping SyncBatchNorm on Mac/CPU—using standard BatchNorm instead.")
        if getattr(args, 'distributed', False):
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=device_ids, find_unused_parameters=True)
            teacher_without_ddp = teacher.module
        else:
            teacher_without_ddp = teacher
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    if getattr(args, 'distributed', False):
        student = nn.parallel.DistributedDataParallel(student, device_ids=device_ids, find_unused_parameters=True)
        student_without_ddp = student.module
    else:
        student_without_ddp = student
    teacher_without_ddp.load_state_dict(student_without_ddp.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both TS networks.")

    # ── Reconstruction decoders (MAE-style auxiliary loss) ────────────────────
    use_reconstruction = cfg.get('use_reconstruction', False)
    student_recon = None
    teacher_recon_decoder = None
    if use_reconstruction:
        student_recon = PatchReconDecoder(embed_dim, args.patch_len).to(device)
        teacher_recon_decoder = copy.deepcopy(student_recon)
        for p in teacher_recon_decoder.parameters():
            p.requires_grad = False
        print("Reconstruction decoders created.")

#-----------------Loss function --------------------
    dino_loss = DINOLoss(
        args.out_dim,
        len(cfg['global_crops']) + len(cfg['local_crops']),
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).to(device)
# ----------------Optimizer --------------------
    params_groups = utils.get_params_groups(student)
    if use_reconstruction:
        # Add reconstruction decoder params (no weight decay on bias/norm, but simpler: just add all)
        params_groups.append({'params': student_recon.parameters()})
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
       # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
#-----------------Scheduler --------------------
    lr_schedule = utils.cosine_scheduler(
        args.lr* (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher,
        1,
        args.epochs,
        len(data_loader),
    )
#----------------Train Loop --------------------
    start_epoch = 0
    best_val_loss = float('inf')

    start_time = time.time()
    print("Starting TS - DINO training !")

    for epoch in range(start_epoch, args.epochs):
        print(f'Starting epoch {epoch}/{args.epochs}')
        if hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            epoch,
            fp16_scaler,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            args,
            student_recon=student_recon,
            teacher_recon_decoder=teacher_recon_decoder,
        )
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if use_reconstruction and student_recon is not None:
            save_dict['student_recon'] = student_recon.state_dict()
            save_dict['teacher_recon_decoder'] = teacher_recon_decoder.state_dict()
        #utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch}.pth'))

        # ── validation + best model ───────────────────────────────────────────
        if val_loader is not None and utils.is_main_process():
            student.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = [s.to(device, non_blocking=True) for s in val_batch]
                    _n_global = len(cfg['global_crops'])
                    _n_dino   = _n_global + len(cfg['local_crops'])
                    dino_val = val_batch[:_n_dino]
                    teacher_out = teacher(dino_val[:_n_global])
                    student_out = student(dino_val)
                    v_loss = dino_loss(student_out, teacher_out, epoch)
                    val_loss_sum += v_loss.item() * val_batch[0].size(0)
                    val_n += val_batch[0].size(0)
            val_loss = val_loss_sum / max(val_n, 1)
            print(f"Epoch {epoch} — val loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                print(f"  → New best val loss: {best_val_loss:.6f} — saved checkpoint_best.pth")
            student.train()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader, optimizer, epoch, fp16_scaler, lr_schedule, wd_schedule, momentum_schedule, args, student_recon=None, teacher_recon_decoder=None):
    student_without_ddp = student.module if hasattr(student, 'module') else student
    student.train()
    teacher.train()  # teacher is in eval mode but we need to keep track of BN stats
    if student_recon is not None:
        student_recon.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('weight_decay', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n_global = len(cfg['global_crops'])
    n_dino   = n_global + len(cfg['local_crops'])
    recon_loss_weight = cfg.get('recon_loss_weight', 0.5)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = batch
        samples = [s.to(device, non_blocking=True) for s in samples]
        # Separate DINO crops from optional reconstruction pair
        dino_samples   = samples[:n_dino]
        use_recon_this = (student_recon is not None) and (len(samples) == n_dino + 2)
        # update learning rate and weight decay according to their schedule
        it_global = it + epoch * len(data_loader)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it_global]
            if i == 0:  # only the first group is regularized
                param_group['weight_decay'] = wd_schedule[it_global]
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(weight_decay=optimizer.param_groups[0]['weight_decay'])
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(dino_samples[:n_global])
            student_output = student(dino_samples)
            loss = dino_loss(student_output, teacher_output, epoch)

            # ── Reconstruction loss (MAE-style, original data only) ────────────
            if use_recon_this:
                full_orig   = samples[n_dino]      # [B, seq_len, n_vars]  → teacher input
                masked_orig = samples[n_dino + 1]  # [B, seq_len, n_vars]  → student input

                # Teacher encodes full original (no grad — teacher already frozen)
                with torch.no_grad():
                    t_tokens = teacher_without_ddp.backbone.forward_recon(full_orig)
                    # [B, num_patch, n_vars, d_model]

                # Student encodes masked original
                s_tokens = student_without_ddp.backbone.forward_recon(masked_orig)
                # [B, num_patch, n_vars, d_model]

                B, num_patch, n_vars, d_model = t_tokens.shape
                # Reshape → [B * n_vars, num_patch, d_model] for decoder
                t_flat = t_tokens.permute(0, 2, 1, 3).reshape(B * n_vars, num_patch, d_model)
                s_flat = s_tokens.permute(0, 2, 1, 3).reshape(B * n_vars, num_patch, d_model)

                t_recon = teacher_recon_decoder(t_flat)  # [B*n_vars, num_patch, patch_len]
                s_recon = student_recon(s_flat)           # [B*n_vars, num_patch, patch_len]

                recon_loss = F.mse_loss(s_recon, t_recon.detach())
                loss = loss + recon_loss_weight * recon_loss
                metric_logger.update(recon_loss=recon_loss.item())

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                            args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it_global]  # momentum parameter
            for param_q, param_k in zip(student_without_ddp.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            # EMA update for the reconstruction teacher decoder
            if use_recon_this:
                for param_q, param_k in zip(student_recon.parameters(), teacher_recon_decoder.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        if device.type == 'cuda':
            torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(len(cfg['global_crops']))

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        print('DINO loss:', total_loss.item())
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        batch_center = batch_center / (len(teacher_output) * world_size)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
#------Data Augmentation for Time Series DINO -------
class DataAugmentationDino:
    """
    Config-driven augmentation.  Reads global_crops / local_crops from config.py.
    Each crop spec dict maps directly to a DWTAugmentation instance;
    if 'type' is a list, one type is drawn at random per sample.
    """

    def __init__(self, global_crops, local_crops, dwt_cfg):
        self.global_crops       = global_crops
        self.local_crops        = local_crops
        self.use_reconstruction = dwt_cfg.get('use_reconstruction', False)
        self.recon_mask_ratio   = dwt_cfg.get('recon_mask_ratio',   0.4)
        self.patch_len          = dwt_cfg.get('patch_len',          16)
        # Pre-build one DWTAugmentation per (crop_index, aug_type) pair
        self._transforms = self._build_transforms(global_crops + local_crops, dwt_cfg)

    # Registry of non-DWT transform names → classes in data_agumentation.py
    _NON_DWT_REGISTRY = {
        'polar':            aug.polar_transformation,
        'galilien':         aug.galilien_transformation,
        'rotation':         aug.rotation_transformation,
        'boost':            aug.boost_transformation,
        'lorentz':          aug.lorentz_transformation,
        'hyperbolic_warp':  aug.hyperbolic_amplitude_warp,
        'hyperbolic_geom':  aug.HyperBolicGeometry,
    }

    def _build_transforms(self, all_specs, dwt_cfg):
        transforms = []
        for spec in all_specs:
            types = spec['type'] if isinstance(spec['type'], list) else [spec['type']]
            per_type = {}
            for t in types:
                if t.startswith('dwt_'):
                    mode = t[4:]   # strip leading 'dwt_'
                    _wavelet_pool = spec.get('wavelet_pool', dwt_cfg.get('dwt_wavelet_pool', None))
                    per_type[t] = aug.DWTAugmentation(
                        wavelet                  = spec.get('wavelet',                    dwt_cfg['dwt_wavelet']),
                        wavelet_pool             = _wavelet_pool,
                        level                    = spec.get('level',                      dwt_cfg['dwt_level']),
                        mode                     = mode,
                        soft_threshold_sigma     = spec.get('soft_threshold_sigma',       dwt_cfg.get('dwt_soft_threshold_sigma', 0.3)),
                        zero_out_ratio           = spec.get('zero_out_ratio',             dwt_cfg.get('dwt_zero_out_ratio', 0.3)),
                        finest_levels            = spec.get('finest_levels',              dwt_cfg.get('dwt_finest_levels', 1)),
                        high_perturb_noise_range = spec.get('high_perturb_noise_range',   dwt_cfg.get('dwt_high_perturb_noise_range', (0.03, 0.08))),
                        band_scale_approx_range  = dwt_cfg.get('dwt_band_scale_approx_range', (0.9, 1.1)),
                        band_scale_detail_range  = dwt_cfg.get('dwt_band_scale_detail_range', (0.6, 1.4)),
                    )
                elif t in self._NON_DWT_REGISTRY:
                    cls = self._NON_DWT_REGISTRY[t]
                    # Each class reads its params from the spec first, then falls back to cfg defaults
                    kwargs = {}
                    if t == 'lorentz':
                        kwargs['v_range']        = spec.get('v_range',        dwt_cfg.get('lorentz_v_range',        (0.2, 0.6)))
                    elif t == 'polar':
                        kwargs['warp_range']     = spec.get('warp_range',     dwt_cfg.get('polar_warp_range',       (0.7, 1.3)))
                    elif t == 'galilien':
                        kwargs['a_range']        = spec.get('a_range',        dwt_cfg.get('galilien_a_range',       (0.8, 1.2)))
                    elif t == 'rotation':
                        kwargs['angle_range']    = spec.get('angle_range',    dwt_cfg.get('rotation_angle_range',   (0, 0.3927)))
                    elif t == 'boost':
                        kwargs['b_range']        = spec.get('b_range',        dwt_cfg.get('boost_b_range',          (0.01, 0.3)))
                    elif t == 'hyperbolic_warp':
                        kwargs['warp_range']     = spec.get('warp_range',     dwt_cfg.get('hyperbolic_warp_range',  (0.5, 1.5)))
                    elif t == 'hyperbolic_geom':
                        kwargs['shift_magnitude']= spec.get('shift_magnitude',dwt_cfg.get('hyperbolic_shift_magnitude', 0.3))
                    per_type[t] = cls(**kwargs)
                else:
                    raise ValueError(f"Unknown augmentation type '{t}'. DWT types must start with 'dwt_'. "
                                     f"Non-DWT types: {list(self._NON_DWT_REGISTRY)}")
            transforms.append(per_type)
        return transforms

    def _random_crop(self, x, crop_ratio):
        timesteps = x.shape[0]
        crop_len  = int(timesteps * crop_ratio)
        if crop_len >= timesteps:
            return x
        start = np.random.randint(0, timesteps - crop_len + 1)
        return x[start : start + crop_len, :]

    def _mask_patches(self, x):
        """Randomly zero out recon_mask_ratio fraction of non-overlapping patches.
        x: [seq_len, n_vars] — the ORIGINAL (unaugmented) input.
        Returns masked copy of same shape.
        """
        n_patches = x.shape[0] // self.patch_len
        masked    = x.clone()
        for p in range(n_patches):
            if random.random() < self.recon_mask_ratio:
                masked[p * self.patch_len : (p + 1) * self.patch_len] = 0.0
        return masked

    def __call__(self, x):
        # ── DINO contrastive views (DWT augmented) ────────────────────────────
        crops     = []
        all_specs = self.global_crops + self.local_crops
        for i, spec in enumerate(all_specs):
            crop_ratio = spec.get('crop_ratio', 1.0)
            x_in       = self._random_crop(x, crop_ratio) if crop_ratio < 1.0 else x
            aug_type   = spec['type']
            if isinstance(aug_type, list):
                aug_type = random.choice(aug_type)
            crops.append(self._transforms[i][aug_type](x_in))

        # ── Reconstruction pair (original data only, no DWT) ──────────────────
        if self.use_reconstruction:
            crops.append(x.clone())          # full original  → teacher recon
            crops.append(self._mask_patches(x))  # masked original → student recon

        return crops

#----Test run -----
def test_run(args):
    utils.init_distributed_mode(args)

    # ── PatchTST-identical data loading ──────────────────────────────────────
    _SEQ_LEN = 336   # 21 patches × 16 = same as PatchTST linear-probe context
    _patchtst_dir = str(Path(__file__).parent.parent / "PatchTST_self_supervised")
    if _patchtst_dir not in sys.path:
        sys.path.insert(0, _patchtst_dir)
    from src.data.pred_dataset import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

    _csv_path = args.data_path_forecast_training
    _root = os.path.dirname(os.path.abspath(_csv_path))
    _fname = os.path.basename(_csv_path)
    _size = [_SEQ_LEN, 0, args.pred_len]
    if 'etth' in _fname.lower():
        _DS_CLS = Dataset_ETT_hour
    elif 'ettm' in _fname.lower():
        _DS_CLS = Dataset_ETT_minute
    else:
        _DS_CLS = Dataset_Custom

    dataset_forecasting_train = _DS_CLS(_root, split='train', size=_size, features='M', data_path=_fname)
    try:
        dataset_forecasting_val = _DS_CLS(_root, split='val', size=_size, features='M', data_path=_fname)
    except Exception:
        dataset_forecasting_val = None
    dataset_forecasting_test  = _DS_CLS(_root, split='test',  size=_size, features='M', data_path=_fname)

    _is_distributed = utils.is_dist_avail_and_initialized()
    data_loader_forecasting_train = torch.utils.data.DataLoader(
        dataset_forecasting_train,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset_forecasting_train, shuffle=False) if _is_distributed else torch.utils.data.SequentialSampler(dataset_forecasting_train),
        batch_size=args.batch_size_forecast,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_forecasting_test = torch.utils.data.DataLoader(
        dataset_forecasting_test,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset_forecasting_test, shuffle=False) if _is_distributed else torch.utils.data.SequentialSampler(dataset_forecasting_test),
        batch_size=args.batch_size_forecast,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_forecasting_val = None
    if dataset_forecasting_val is not None:
        data_loader_forecasting_val = torch.utils.data.DataLoader(
            dataset_forecasting_val,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset_forecasting_val, shuffle=False) if _is_distributed else torch.utils.data.SequentialSampler(dataset_forecasting_val),
            batch_size=args.batch_size_forecast,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _forecast_num_patch = _SEQ_LEN // args.patch_len   # 336 // 16 = 21
    model = PatchTST(
        c_in= args.c_in,
        target_dim=args.pred_len,
        patch_len=args.patch_len,
        num_patch=_forecast_num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.embed_dim,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=getattr(args, 'head_dropout_forecasting', args.head_dropout),
        act='leakyrelu',
        head_type='prediction',
        res_attention=False,
        drop_path_rate=args.drop_path_rate,
        step_size=1,
        mlp_head=getattr(args, 'mlp_head', False),
        )
    criterion = nn.MSELoss()
    _lp_fore  = getattr(args, 'linear_probe', True)
    _head_lr_fore = float(args.lr_forecasting)
    _enc_lr       = float(getattr(args, 'lr_forecasting_encoder', None) or _head_lr_fore)
    if _lp_fore:
        optimizer = torch.optim.Adam(model.head.parameters(),
                                     lr=_head_lr_fore, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam([
            {"params": model.head.parameters(),     "lr": _head_lr_fore},
            {"params": model.backbone.parameters(), "lr": _enc_lr},
        ], weight_decay=1e-4)
        print(f"  [DINO forecast] head_lr={_head_lr_fore}  encoder_lr={_enc_lr}")
    model = model.to(device)
    if args.path_num != 0:
        if args.path_num == "best":
            path = os.path.join(args.output_dir, 'checkpoint_best.pth')
        else:
            path = os.path.join(args.output_dir, f'checkpoint{args.path_num}.pth')
        print(f"Loading checkpoint: {path}")
        if not os.path.exists(path):
            print(f"  WARNING: checkpoint not found at {path}, using random init.")
        else:
            checkpoint = torch.load(path, weights_only=False, map_location=device)

            # Use teacher (EMA) weights — better representations than student for downstream.
            # Keys: TSMultiCropWrapper.backbone.* → strip first 'backbone.' to match PatchTST.
            state_dict = checkpoint['teacher']
            new_state_dict = {}
            model_state = model.state_dict()

            for key, value in state_dict.items():
                # Remove 'module.' prefix (DistributedDataParallel)
                new_key = key.replace('module.', '')
                # Strip one 'backbone.' (TSMultiCropWrapper wrapper)
                if new_key.startswith('backbone.'):
                    new_key = new_key[len('backbone.'):]

                # Only load if key exists in forecasting model and shapes match
                if new_key in model_state and model_state[new_key].shape == value.shape:
                    new_state_dict[new_key] = value

            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"✓ Loaded DINO teacher checkpoint from epoch {args.path_num}")
            print(f"  Loaded {len(new_state_dict)} / {len(model_state)} weights")
            print(f"  Missing: {missing}")
            print(f"  Missing (new head): {len(missing)}  |  Unexpected (DINO head): {len(unexpected)}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr_forecasting,
        total_steps=args.epochs_forecasting * len(data_loader_forecasting_train),
        pct_start=0.3, anneal_strategy='cos',
    )
    if _lp_fore:
        for param in model.backbone.parameters():
            param.requires_grad = False
        _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _total     = sum(p.numel() for p in model.parameters())
        print(f"  [DINO forecast] MODE: linear probe — backbone FROZEN")
        print(f"  Trainable: {_trainable:,} / {_total:,} params")
    else:
        _total = sum(p.numel() for p in model.parameters())
        print(f"  [DINO forecast] MODE: full fine-tuning — backbone UNFROZEN")
        print(f"  Trainable: {_total:,} / {_total:,} params")

    best_val_loss_fc = float('inf')
    best_state_fc    = None
    for epoch in range(args.epochs_forecasting):
        model.train()
        for it, batch in enumerate(data_loader_forecasting_train):
            samples, labels = batch
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(samples)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # ── val MSE + keep-best ──
        if data_loader_forecasting_val is not None:
            model.eval()
            v_sum = 0.0; v_n = 0
            with torch.no_grad():
                for v_batch in data_loader_forecasting_val:
                    v_x, v_y = v_batch
                    v_x = v_x.to(device, non_blocking=True)
                    v_y = v_y.to(device, non_blocking=True)
                    v_out = model(v_x)
                    v_sum += criterion(v_out, v_y).item() * v_x.size(0)
                    v_n   += v_x.size(0)
            val_loss = v_sum / max(v_n, 1)
            if val_loss < best_val_loss_fc:
                best_val_loss_fc = val_loss
                best_state_fc    = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  [DINO forecast] epoch {epoch} — val MSE: {val_loss:.6f}  (best: {best_val_loss_fc:.6f})")

    if best_state_fc is not None:
        model.load_state_dict(best_state_fc)
        print(f"  [DINO forecast] Restored best checkpoint (best val MSE={best_val_loss_fc:.6f})")
    # Testing
    model.eval()
    os.makedirs("test_results/tests", exist_ok=True)
    total_loss = 0.0
    count = 0
    model.operation = 'test'

    # Initialize accumulators for metrics across ALL test batches
    num_vars = args.c_in
    accumulated_mse = torch.zeros(num_vars).to(device)
    accumulated_mae = torch.zeros(num_vars).to(device)
    num_samples = 0

    with torch.no_grad():
        txt_save_path = f"test_results/tests/metrics_results_{args.path_num}.txt"
        first_batch_saved = False

        # Loop through ALL test batches
        for it, batch in enumerate(data_loader_forecasting_test):
            samples, labels = batch
            samples = samples.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            outputs = model(samples)

            # Compute MSE and MAE for this batch
            batch_size = outputs.shape[0]
            squared_errors = (outputs - labels) ** 2
            absolute_errors = torch.abs(outputs - labels)

            # Average over batch and time dimension, keep variable dimension
            batch_mse = squared_errors.mean(dim=(0, 1))  # [num_vars]
            batch_mae = absolute_errors.mean(dim=(0, 1))  # [num_vars]

            # Accumulate (weighted by batch size)
            accumulated_mse += batch_mse * batch_size
            accumulated_mae += batch_mae * batch_size
            num_samples += batch_size

            # Save visualization for first batch only (skip for large multivariate datasets)
            if not first_batch_saved:
                n_vars = outputs.shape[-1]
                if n_vars > 20:
                    first_batch_saved = True  # skip plotting
                    continue
                fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)
                if n_vars == 1: axes = [axes]

                for v in range(n_vars):
                    var_name = args.parms_for_testing_forecasting[v] if hasattr(args, 'parms_for_testing_forecasting') else f"Var {v}"

                    truth_segment = labels[0, :, v].cpu().numpy()
                    pred_segment  = outputs[0, :, v].cpu().numpy()
                    x = np.arange(len(truth_segment))

                    axes[v].plot(x, truth_segment, label="Ground Truth", color="black", linewidth=1.5)
                    axes[v].plot(x, pred_segment,  label="Forecast",     color="red",   linestyle="--", alpha=0.8)

                    axes[v].set_title(f"Variable {v}: {var_name}", fontsize=14, loc='left')
                    axes[v].legend(loc="upper left")
                    axes[v].grid(True, alpha=0.2)
                    axes[v].set_ylabel("Value")

                plt.xlabel("Forecast Time Steps")
                plt.tight_layout()

                save_path = f"test_results/tests/full_multivariable_forecast_{args.path_num}.png"
                plt.savefig(save_path)
                print(f"✅ Figure saved to: {save_path}")
                plt.close()
                first_batch_saved = True

        # Compute final averages across ALL test samples
        final_mse = (accumulated_mse / num_samples).cpu().numpy()
        final_mae = (accumulated_mae / num_samples).cpu().numpy()
        final_rmse = np.sqrt(final_mse)

        # Save metrics to file
        with open(txt_save_path, "w") as f:
            f.write(f"TEST SET METRICS (Averaged over {num_samples} samples)\n")
            f.write(f"Path: {args.path_num} (0=random, >0=DINO checkpoint)\n")
            f.write("="*60 + "\n\n")
            f.write(f"{'Var_Idx':<10} | {'MSE':<12} | {'MAE':<12} | {'RMSE':<12}\n")
            f.write("-" * 60 + "\n")

            for v in range(num_vars):
                f.write(f"{v:<10} | {final_mse[v]:<12.6f} | {final_mae[v]:<12.6f} | {final_rmse[v]:<12.6f}\n")

            f.write("\n" + "="*60 + "\n")
            f.write(f"OVERALL AVERAGES:\n")
            f.write(f"  Mean MSE:  {final_mse.mean():.6f}\n")
            f.write(f"  Mean MAE:  {final_mae.mean():.6f}\n")
            f.write(f"  Mean RMSE: {final_rmse.mean():.6f}\n")

        print(f"\n{'='*60}")
        print(f"TEST RESULTS (path_num={args.path_num}):")
        print(f"{'='*60}")
        print(f"Mean MSE:  {final_mse.mean():.6f}")
        print(f"Mean MAE:  {final_mae.mean():.6f}")
        print(f"Mean RMSE: {final_rmse.mean():.6f}")
        print(f"✅ Detailed metrics saved to {txt_save_path}")
        print(f"{'='*60}\n")
        return float(final_mse.mean())
        
def train_classification(args, classification_train=None, classification_val=None,
                         classification_test=None, n_classes=None):
    """Train classification head on top of pretrained DINO encoder.

    If classification_train/val/test DataLoaders are provided they are used directly
    (elastic ClassificationDataPuller path). Otherwise falls back to the legacy
    DataPullerClassification path for backwards compatibility.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training classification on {device}")

    if classification_train is not None:
        # ── elastic path: callers supply pre-built DataLoaders ────────────────
        train_loader = classification_train
        val_loader   = classification_val
        test_loader  = classification_test
        # infer shape from first batch (batch is (x, y, padding_mask) 3-tuple)
        _batch = next(iter(train_loader))
        _x = _batch[0]  # [B, n_patches, patch_size, n_vars] or [B, T, C]
        n_vars = _x.shape[-1]
        if _x.dim() == 3:   # raw (B, T, C) from UEADataset
            num_patch = _x.shape[1] // args.patch_len
        else:               # pre-patched (B, P, PL, C) from ClassificationDataPuller
            num_patch = _x.shape[1]
        n_cls     = n_classes
    else:
        # ── legacy UCI HAR path ───────────────────────────────────────────────
        train_dataset = dpuller.DataPullerClassification(
            data_dir=args.data_path_classification,
            split='train',
            c_in=args.c_in_classification,
            seq_len=args.seq_len_classification,
            normalize=True
        )
        test_dataset = dpuller.DataPullerClassification(
            data_dir=args.data_path_classification,
            split='test',
            c_in=args.c_in_classification,
            seq_len=args.seq_len_classification,
            normalize=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size_classification,
            shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader   = None
        test_loader  = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size_classification,
            shuffle=False, num_workers=args.num_workers, pin_memory=True)
        n_vars    = args.c_in_classification
        num_patch = args.seq_len_classification // args.patch_len
        n_cls     = args.n_classes

    # Create model with classification head
    model = PatchTST(
        c_in=n_vars,
        target_dim=n_cls,
        patch_len=args.patch_len,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.embed_dim,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='gelu',
        head_type='classification',
        res_attention=False,
        step_size=args.patch_len,
        mlp_head=getattr(args, 'mlp_head', False),
    ).to(device)

    # Load pretrained encoder
    if args.path_num == "best":
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
    elif isinstance(args.path_num, str) and os.path.isabs(args.path_num):
        checkpoint_path = args.path_num   # direct absolute file path (TEMP: for local testing, remove after)
    elif args.path_num and int(args.path_num) > 0:
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint{args.path_num}.pth')
    else:
        checkpoint_path = None

    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['student']
            model_dict = model.state_dict()
            new_state_dict = {}
            for key, value in state_dict.items():
                # Remove 'module.' prefix (from DistributedDataParallel)
                new_key = key.replace('module.', '')
                # Remove first 'backbone.' (from TSMultiCropWrapper)
                if new_key.startswith('backbone.'):
                    new_key = new_key.replace('backbone.', '', 1)
                # Skip DINO head params, only load encoder
                if 'head' in key.split('.')[-2:]:
                    continue
                if new_key in model_dict and model_dict[new_key].shape == value.shape:
                    new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded encoder from checkpoint {args.path_num}")
            print(f"Loaded {len(new_state_dict)}/{len(model_dict)} parameters")
        else:
            print(f"Checkpoint {checkpoint_path} not found, using random initialization")

    # Freeze encoder (linear probe) or fine-tune full model
    _lp_cls = getattr(args, 'linear_probe', True)
    for name, param in model.named_parameters():
        if _lp_cls:
            param.requires_grad = ('head' in name)
        else:
            param.requires_grad = True
    _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _total     = sum(p.numel() for p in model.parameters())
    if _lp_cls:
        print(f"  [DINO classify] MODE: linear probe — encoder FROZEN")
    else:
        print(f"  [DINO classify] MODE: full fine-tuning — encoder UNFROZEN")
    print(f"  Trainable: {_trainable:,} / {_total:,} params")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    _head_lr_cls = float(args.lr_classification)
    _enc_lr_cls  = float(getattr(args, 'lr_classification_encoder', None) or _head_lr_cls)
    if _lp_cls:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=_head_lr_cls,
        )
    else:
        head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
        enc_params  = [p for n, p in model.named_parameters() if "head" not in n and p.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": head_params, "lr": _head_lr_cls},
            {"params": enc_params,  "lr": _enc_lr_cls},
        ])
        print(f"  [DINO classify] head_lr={_head_lr_cls}  encoder_lr={_enc_lr_cls}")

    # Cosine learning rate scheduler
    lr_schedule = utils.cosine_scheduler(
        base_value=args.lr_classification,
        final_value=args.min_lr_classification,
        epochs=args.epochs_classification,
        niter_per_ep=len(train_loader),
        warmup_epochs=10,
    )

    print(f"Starting classification training for {args.epochs_classification} epochs...")

    best_acc   = 0.0
    best_state = None
    for epoch in range(args.epochs_classification):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for it, batch in enumerate(train_loader):
            inputs, labels = batch[0], batch[1]
            padding_mask = batch[2] if len(batch) > 2 else None
            # Update learning rate
            step = epoch * len(train_loader) + it
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[step]

            inputs, labels = inputs.to(device), labels.to(device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            labels = labels.squeeze(-1) if labels.dim() > 1 else labels
            if inputs.dim() == 4:
                B, P, PL, C = inputs.shape
                inputs = inputs.reshape(B, P * PL, C)

            # Forward pass
            outputs = model(inputs, padding_mask=padding_mask)  # [batch_size, n_classes]
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss /= len(train_loader)
        train_acc = correct / total

        # Use val_loader for best-model selection if available, else test_loader
        eval_loader = val_loader if val_loader is not None else test_loader
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                inputs, labels = batch[0], batch[1]
                padding_mask = batch[2] if len(batch) > 2 else None
                inputs, labels = inputs.to(device), labels.to(device)
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device)
                if inputs.dim() == 4:
                    B, P, PL, C = inputs.shape
                    inputs = inputs.reshape(B, P * PL, C)
                outputs = model(inputs, padding_mask=padding_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_loss /= len(eval_loader)
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{args.epochs_classification}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'val_acc': val_acc
            }
            _ckpt_tag = args.path_num if isinstance(args.path_num, int) else 'custom'
            save_path = os.path.join(args.output_dir, f'classification_best_checkpoint{_ckpt_tag}.pth')
            torch.save(save_dict, save_path)

    # Final test accuracy with best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0], batch[1]
            padding_mask = batch[2] if len(batch) > 2 else None
            inputs, labels = inputs.to(device), labels.to(device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            if inputs.dim() == 4:
                B, P, PL, C = inputs.shape
                inputs = inputs.reshape(B, P * PL, C)
            outputs = model(inputs, padding_mask=padding_mask)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = correct / total

    print(f"\n[DINO] Classification complete! Test Accuracy: {test_acc:.4f}  (best val: {best_acc:.4f})")
    return test_acc
