import copy
import math
import os

import torch
import torch.nn.functional as F

from mask_util import apply_mask

def _instance_norm(x, eps=1e-6):
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2), keepdim=True) + eps
    return (x - mean) / std, mean, std


def compute_discrete_jepa_loss(
    self,
    context_out,
    target_out,
    masks,
    non_masks,
    epoch,
    patches=None,
    lambda_weights={'s2p': 1.0, 'p2s': 1.0, 'p2p': 1.0},
    beta_vq=1.0,
    current_global_step=0,
    total_training_steps=100000,
    vq_warmup=0.15,
    batch_idx=0
):
    z_s_target = target_out["quantized_semantic"]
    z_p_target = target_out["data_patches"]
    z_s_context = context_out["quantized_semantic"]
    z_p_context = context_out["data_patches"]
# fix z_s
    # Apply VQ consistently (removed conditional epoch logic)
    # Context: Student VQ learns from student encoder (gradients flow to codebook)
    l_vq, z_s_context, perplexity, indices, encodings_context, soft_avg_probs = self.vector_quantizer(z_s_context)
    # Target: EMA VQ (teacher codebook, updated via momentum) on detached EMA encoder output
    _, z_s_target, _, _, _, _ = self.vector_quantizer(z_s_target.detach())

    pred_s2p = self.predictor(z_s_context, target_mask=non_masks, task='S2P')
    l_s2p = F.mse_loss(pred_s2p, z_p_target.detach())
    pred_p2s = self.predictor(z_p_context, target_mask=non_masks, task='P2S')
    l_p2s = F.mse_loss(pred_p2s, z_s_target.detach())
    pred_p2p = self.predictor(z_p_context, target_mask=non_masks, task='P2P')
    l_p2p = F.mse_loss(pred_p2p, z_p_target.detach())
    # Use soft (differentiable) assignments for perplexity loss so gradients
    # flow to both encoder outputs and codebook weights
    diff_entropy = -torch.sum(soft_avg_probs * torch.log(soft_avg_probs + 1e-10))
    l_preplexity = torch.log(torch.tensor(float(self.config["codebook_size"]), device=soft_avg_probs.device)) - diff_entropy
    active_codes = (encodings_context.sum(0) > 0).sum()
    usage_pct = active_codes.float() / self.vector_quantizer._num_embeddings * 100
    var_loss_context_patch, cov_loss_context_patch = self._calculate_vicreg_loss(z_p_context)
    var_loss_context_token, cov_loss_context_token = context_out["var_loss"], context_out["covar_loss"]
    #token_div_loss = self._calculate_token_diversity_loss(z_s_context)
    #cross_decorr_loss = self._cross_decorrelation_loss(z_p_context, z_s_context)
    #grounding_loss = self._grounding_loss(pred_p2p, patches, non_masks) if patches is not None else torch.tensor(0.0, device=z_p_context.device)
    total_loss = (
        1.0*(lambda_weights["S2P"] * l_s2p +
        lambda_weights["P2P"] * l_p2p +
        lambda_weights["P2S"] *l_p2s)+
        beta_vq * l_vq +
        self.config["preplexity_coeff"] * l_preplexity +
        #self.config["token_diversity"] * token_div_loss +
        self.config["vigreg_var"] * (var_loss_context_patch) +
        self.config["vigreg_covar"] * (cov_loss_context_patch)
        #+ self.config["decorr_coeff"] * cross_decorr_loss
        #+ self.config["grounding_coeff"] * grounding_loss
    )
    if batch_idx % 5 == 0:
        print(f"TOTAL: {total_loss.item():.4f} | P2P: {l_p2p.item():.4f}, S2P: {l_s2p.item():.4f}, P2S: {l_p2s.item():.4f}, VQ: {l_vq.item():.4f}, Perp: {l_preplexity:.4f}")
        print(f"  var[patch={var_loss_context_patch.item():.4f} tok={var_loss_context_token.item():.4f}] cov[patch={cov_loss_context_patch.item():.4f} tok={cov_loss_context_token.item():.4f}] | codes={active_codes} ({usage_pct:.1f}%)")

    return total_loss, {
        'l_s2p': l_s2p.item(),
        'l_p2p': l_p2p.item(),
        'l_p2s': l_p2s.item(),
        'l_vq': l_vq.item(),
        'l_preplexity': l_preplexity,
        'var_loss_context_patch': var_loss_context_patch.item(),
        'var_loss_context_token': var_loss_context_token.item(),
        'cov_loss_context_patch': cov_loss_context_patch.item(),
        'cov_loss_context_token': cov_loss_context_token.item(),
    }


def evaluate(self, val_loader, lambda_weights, beta_vq, current_global_step, total_training_steps, vq_warmup, epoch):
    self.encoder.eval()
    self.encoder_ema.eval()
    self.predictor.eval()
    self.vector_quantizer.eval()  # Disable EMA codebook updates during validation
    val_loss = 0.0
    val_metrics = {'l_s2p': 0.0, 'l_p2s': 0.0, 'l_p2p': 0.0, 'l_vq': 0.0, 'l_preplexity': 0.0, 'var_loss_context_patch': 0.0,
    'var_loss_context_token': 0.0, 'cov_loss_context_patch': 0.0, 'cov_loss_context_token': 0.0}
    with torch.no_grad():
        for patches, masks, non_masks in val_loader:
            patches, masks, non_masks = patches.to(self.device), masks.to(self.device), non_masks.to(self.device)
            # masks=context_idx (visible patches), non_masks=target_idx (hidden patches to predict)
            #patches, _, _ = _instance_norm(patches)
            target_out = self.encoder_ema(patches)
            target_out["data_patches"] = apply_mask(target_out["data_patches"], non_masks)         # EMA: keep hidden (target) patches
            context_out = self.encoder(patches, mask=masks)        # student: see visible (context) patches
            loss, loss_dict = self.compute_discrete_jepa_loss(
                context_out,
                target_out,
                masks,
                non_masks,
                epoch,
                patches=patches,
                lambda_weights=lambda_weights,
                beta_vq=beta_vq,
                current_global_step=current_global_step,
                total_training_steps=total_training_steps,
                vq_warmup=vq_warmup
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
        "vector_quantizer": self.vector_quantizer.state_dict(),
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
    self.vector_quantizer = self.vector_quantizer.to(self.device)
    for p in self.encoder_ema.parameters():
        p.requires_grad = False

    num_batches = self.steps_per_epoch
    best_val_loss = float("inf")
    best_val_pred_loss = float("inf")   # P2P + S2P + P2S only — used for model selection
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
        self.vector_quantizer.train()  # Enable EMA codebook updates during training
        running_loss = 0.0

        for batch_idx, (patches, masks, non_masks) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            m = next(self.ema_scheduler)
            # Cosine weight decay schedule: ramp wd from start → end over total steps
            wd_factor = 0.5 * (1 - math.cos(math.pi * current_global_step / max(self.total_steps, 1)))
            self.optimizer.param_groups[0]["weight_decay"] = wd_start + wd_factor * (wd_end - wd_start)
            self.optimizer.param_groups[1]["weight_decay"] = wd_pred_start + wd_factor * (wd_pred_end - wd_pred_start)
            # param_groups[2] is codebook — weight_decay stays 0
            patches = patches.to(self.device)
            masks = masks.to(self.device)
            non_masks = non_masks.to(self.device)
            #patches, _, _ = _instance_norm(patches)
            with torch.no_grad():
                target_out = self.encoder_ema(patches)
                target_out["data_patches"] = apply_mask(target_out["data_patches"], non_masks)   # EMA keeps target (masked) patches
                z_s_target = target_out["quantized_semantic"]
                z_p_target = target_out["data_patches"]

            context_out = self.encoder(patches, mask=masks)       # student encoder sees context patches
            loss, loss_dict = self.compute_discrete_jepa_loss(
                context_out,
                target_out,
                masks,
                non_masks,
                epoch,
                patches=patches,
                lambda_weights=self.config["lambda_weights"],
                beta_vq=self.config["beta_vq"],
                current_global_step=current_global_step,
                total_training_steps=self.total_steps,
                vq_warmup=self.config["vq_warmup"],
                batch_idx=batch_idx
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config["clip_grad"])
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.config["clip_grad"])
            torch.nn.utils.clip_grad_norm_(self.vector_quantizer.parameters(), self.config["clip_grad"])
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
        val_loss, val_dict = self.evaluate(self.val_loader, self.config["lambda_weights"], self.config["beta_vq"], current_global_step, self.total_steps, self.config["vq_warmup"], epoch)

        # Save Best Model
        if val_loss < best_val_loss and epoch >= self.warmup:
            best_val_loss = val_loss
            self.save_model(self.encoder, self.encoder_ema, self.predictor, self.optimizer, epoch, f"{self.path_save}")
            self.best_model = {
                "encoder": copy.deepcopy(self.encoder.state_dict()),
                "predictor": copy.deepcopy(self.predictor.state_dict()),
                "encoder_ema": copy.deepcopy(self.encoder_ema.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "vector_quantizer": copy.deepcopy(self.vector_quantizer.state_dict()),
                "epoch": epoch
            }
            print("New best validation loss! Model saved.")
        if epoch%1==0:
            self.save_model(self.encoder, self.encoder_ema, self.predictor, self.optimizer, epoch, f"{self.path_save}_epoch{epoch}")
            self.best_model = {
                "encoder": copy.deepcopy(self.encoder.state_dict()),
                "predictor": copy.deepcopy(self.predictor.state_dict()),
                "encoder_ema": copy.deepcopy(self.encoder_ema.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "vector_quantizer": copy.deepcopy(self.vector_quantizer.state_dict()),
                "epoch": epoch
            }
            print("saved at epoch")

    print("Training complete. Starting Final Test:")
    test_loss, test_dict = self.evaluate(self.test_loader, self.config["lambda_weights"], self.config["beta_vq"], current_global_step, self.total_steps, self.config["vq_warmup"], 101)
    print(f"FINAL TEST RESULTS | Loss: {test_loss:.4f} | S2P: {test_dict['l_s2p']:.4f}")
    return test_loss, test_dict, self.best_model
