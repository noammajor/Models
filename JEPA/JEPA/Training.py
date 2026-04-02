import copy
import math
import os

import torch
import torch.nn.functional as F

from mask_util import apply_mask

def _instance_norm(x, eps=1e-6):
    """Per-instance, per-variable RevIN normalization over the time axis.

    x: [B, n_patches, patch_size, n_vars]
    returns: (x_norm, mean, std)  — mean/std shape [B, 1, 1, n_vars]
    """
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2), keepdim=True) + eps
    return (x - mean) / std, mean, std


def compute_jepa_loss(
    self,
    context_out,
    target_out,
    masks,
    non_masks,
    epoch,
    patches=None,
    current_global_step=0,
    total_training_steps=100000,
    batch_idx=0
):
    z_p_target = target_out["data_patches"]
    z_p_context = context_out["data_patches"]
    pred= self.predictor(z_p_context, target_mask=non_masks)
    l_MSE = F.mse_loss(pred, z_p_target.detach())

    
    
    var_loss_context_patch,var_loss_batch, cov_loss_context_patch = self._calculate_vicreg_loss(z_p_context)
    total_loss = (
        self.config["invar"]*l_MSE +self.config["vigreg_var"] * var_loss_context_patch+ self.config["vigreg_cov"] * cov_loss_context_patch
    )
    if batch_idx % 5 == 0:
        print(f"Epoch {epoch}, Batch {batch_idx} - JEPA Loss: {total_loss.item():.4f}, MSE: {l_MSE.item():.4f}, Var: {var_loss_context_patch.item():.4f}, Var_batch: {var_loss_batch.item():.4f}, Cov: {cov_loss_context_patch.item():.4f}")

    return total_loss, {
        "l_MSE": l_MSE.item(),
        "var_loss_context_patch": var_loss_context_patch.item(),
        "cov_loss_context_patch": cov_loss_context_patch.item()
    }


def evaluate(self, val_loader, current_global_step, total_training_steps, epoch):
    self.encoder.eval()
    self.encoder_ema.eval()
    self.predictor.eval()
    val_loss = 0.0
    val_metrics = {'l_MSE': 0.0, 'var_loss_context_patch': 0.0, 'cov_loss_context_patch': 0.0}
    with torch.no_grad():
        for patches, masks, non_masks in val_loader:
            patches, masks, non_masks = patches.to(self.device), masks.to(self.device), non_masks.to(self.device)
            target_out = self.encoder_ema(patches)
            target_out["data_patches"] = apply_mask(target_out["data_patches"], non_masks)         # EMA: keep hidden (target) patches
            context_out = self.encoder(patches, mask=masks)        # student: see visible (context) patches
            loss, loss_dict = self.compute_jepa_loss(
                context_out,
                target_out,
                masks,
                non_masks,
                epoch,
                patches=patches,
                current_global_step=current_global_step,
                total_training_steps=total_training_steps,
            )
            val_loss += loss.item()
            for k, v in loss_dict.items():
                val_metrics[k] += v
    return val_loss / len(val_loader), {k: v / len(val_loader) for k, v in val_metrics.items()}


def save_model(self, encoder, target_encoder, predictor, optimizer, epoch, path_save):
    checkpoint_dir = os.path.dirname(path_save)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Created directory: {checkpoint_dir}")
        except Exception as e:
            print(f"Could not create directory {checkpoint_dir}: {e}")
            return  # Exit if we can't create the folder

    save_dict = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
    }

    try:
        path_name = f"{path_save}best_model.pt"
        print(f"Saving checkpoint to: {path_name}")
        torch.save(save_dict, path_name)
        print(f"Checkpoint saved: {path_name}")
    except Exception as e:
        print(f"Problem saving checkpoint: {e}")


def train_and_evaluate(self):
    self.encoder = self.encoder.to(self.device)
    self.predictor = self.predictor.to(self.device)
    self.encoder_ema = self.encoder_ema.to(self.device)
    for p in self.encoder_ema.parameters():
        p.requires_grad = False

    num_batches = self.steps_per_epoch
    best_val_loss = float("inf")
    best_val_pred_loss = float("inf")
    total_loss, total_var_encoder, total_var_decoder = 0.0, 0.0, 0.0
    self.save_model(self.encoder, self.encoder_ema, self.predictor, self.optimizer, 0, f"{self.path_save}_INITIAL")
    current_global_step = 0
    wd_start     = self.config["weight_decay"]
    wd_end       = self.config.get("weight_decay_end", wd_start * 10)
    wd_pred_start = self.config["weight_decay_pred"]
    wd_pred_end   = self.config.get("weight_decay_pred_end", wd_pred_start * 10)
    # Training Loop
    for epoch in range(self.config["num_epochs"]):
        print(f"Starting Epoch {epoch}/{self.config['num_epochs']}")
        self.encoder.train()
        self.predictor.train()
        running_loss = 0.0

        for batch_idx, (patches, masks, non_masks) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            m = next(self.ema_scheduler)
            # Cosine weight decay schedule: ramp wd from start → end over total steps
            wd_factor = 0.5 * (1 - math.cos(math.pi * current_global_step / max(self.total_steps, 1)))
            self.optimizer.param_groups[0]["weight_decay"] = wd_start + wd_factor * (wd_end - wd_start)
            self.optimizer.param_groups[1]["weight_decay"] = wd_pred_start + wd_factor * (wd_pred_end - wd_pred_start)
            patches = patches.to(self.device)
            masks = masks.to(self.device)
            non_masks = non_masks.to(self.device)
            with torch.no_grad():
                target_out = self.encoder_ema(patches)
                target_out["data_patches"] = apply_mask(target_out["data_patches"], non_masks)   
                z_p_target = target_out["data_patches"]

            context_out = self.encoder(patches, mask=masks)       # student encoder sees context patches
            loss, loss_dict = self.compute_jepa_loss(
                context_out,
                target_out,
                masks,
                non_masks,
                epoch,
                patches=patches,
                current_global_step=current_global_step,
                total_training_steps=self.total_steps,
                batch_idx=batch_idx
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config["clip_grad"])
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.config["clip_grad"])
            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                for p, p_ema in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
                    p_ema.data.mul_(m).add_((1.0 - m) * p.detach().data)
                    
            running_loss += loss.item()

        epoch_avg_loss = running_loss / len(self.train_loader)
        total_loss += epoch_avg_loss

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, lr: {self.optimizer.param_groups[0]['lr']:.3g} - JEPA Loss: {total_loss:.4f},")
        print(f"Validating set of Epoch: {epoch}")
        val_loss, val_dict = self.evaluate(self.val_loader,  current_global_step, self.total_steps,  epoch)

        # Save Best Model
        if val_loss < best_val_loss and epoch >= self.warmup:
            best_val_loss = val_loss
            self.save_model(self.encoder, self.encoder_ema, self.predictor, self.optimizer, epoch, f"{self.path_save}")
            self.best_model = {
                "encoder": copy.deepcopy(self.encoder.state_dict()),
                "predictor": copy.deepcopy(self.predictor.state_dict()),
                "encoder_ema": copy.deepcopy(self.encoder_ema.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "epoch": epoch
            }
            print("New best validation loss! Model saved.")
        if epoch%1 == 0:
            self.save_model(self.encoder, self.encoder_ema, self.predictor, self.optimizer, epoch, f"{self.path_save}_epoch{epoch}")
            self.best_model = {
                "encoder": copy.deepcopy(self.encoder.state_dict()),
                "predictor": copy.deepcopy(self.predictor.state_dict()),
                "encoder_ema": copy.deepcopy(self.encoder_ema.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "epoch": epoch
            }
            print("saved at epoch")

    print("Training complete. Starting Final Test:")
    test_loss, test_dict = self.evaluate(self.test_loader,  current_global_step, self.total_steps,  101)
    print(f"FINAL TEST RESULTS | Loss: {test_loss:.4f} | MSE: {test_dict['l_MSE']:.4f} | Var: {test_dict['var_loss_context_patch']:.4f} | Cov: {test_dict['cov_loss_context_patch']:.4f}")
    return test_loss, test_dict, self.best_model
