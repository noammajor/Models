import copy
import math

import torch
import torch.nn as nn

from JEPA.Encoder import Encoder
from JEPA.Predictors import JEPAPredictor as Predictor
from JEPA.Decoder import LinearDecoder

from JEPA.Forecasting import (
    forcasting_zeroshot,
)
from JEPA.Classification import (
    classification_zeroshot,
)
from JEPA.Anomaly import (
    anomaly_zeroshot,
)
from JEPA.losses import (
    _calculate_vicreg_loss,
)
from JEPA.Training import (
    compute_jepa_loss,
    evaluate,
    save_model,
    train_and_evaluate,
)

class JEPA(nn.Module):
    def __init__(self, config, input_dim, num_patches, steps_per_epoch, train_loader, val_loader, test_loader, forcasting_train, forcasting_val, forcasting_test):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.encoder = Encoder(
            num_patches=config["ratio_patches"],
            dim_in=input_dim,
            embed_dim=config["encoder_embed_dim"],
            nhead=config["nhead"],
            num_layers=config["num_encoder_layers"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"],
            attn_drop_rate=config["attn_drop_rate"],
            pe='sincos',
            learn_pe=False,
            res_attention=True,
        )
        self.predictor = Predictor(
            num_patches=config["ratio_patches"],
            embed_dim=config["encoder_embed_dim"],
            nhead=config["predictor_nhead"],
            num_layers=config["predictor_num_layers"],
            config=config,
        )


        encoder_params = list(self.encoder.parameters())
        other_params_pred = list(self.predictor.parameters())

        # Switched to AdamW for better transformer training stability
        self.optimizer = torch.optim.SGD([
            {
                "params": encoder_params,
                "lr": config["lr"],
                "weight_decay": config["weight_decay"],
                "momentum": 0.9,
                "nesterov": True
            },
            {
                "params": other_params_pred,
                "lr": config["lr_pred"],
                "weight_decay": config["weight_decay_pred"],
                "momentum": 0.9,
                "nesterov": True
            }
        ])
        self.steps_per_epoch = len(train_loader) if train_loader is not None else 1
        self.total_steps = self.config["num_epochs"] * self.steps_per_epoch

        # mimicing the D-JEPA paper
        #self.scheduler = lr_scheduler.OneCycleLR(
        #self.optimizer,
        #max_lr=self.config["lr"],             # The peak learning rate from your config
        #total_steps=self.total_steps,
        #pct_start=0.05,                  # 5% warmup as per TD-JEPA
        #anneal_strategy='cos',           # Cosine decay is standard used in D-JEPA]
        #div_factor=10.0,                 # changed from 25 to 10
        #final_div_factor=1e4             # defualt
        #)
        if self.train_loader is not None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[config["lr"], config["lr_pred"]],
            epochs=config["num_epochs"],
            steps_per_epoch=self.steps_per_epoch,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
            )
        else:
            self.scheduler = None
        
        self.encoder_ema = copy.deepcopy(self.encoder)
        self.encoder_ema.jepa = True
        self.encoder_ema.type_enc = "target"
        self.encoder_ema.eval()  # Target encoder always deterministic — no dropout
        self.checkpoint_save = self.config["checkpoint_save"]
        self.checkpoint_print = self.config["checkpoint_print"]
        self.path_save = self.config["path_save"]
        self.clip_grad = self.config["clip_grad"]
        self.warmup = self.config["warmup_ratio"] * self.config["num_epochs"]
        _T = int(self.config["num_epochs"] * self.steps_per_epoch)
        _m0 = self.config["ema_momentum"]
        self.ema_scheduler = (
            1 - (1 - _m0) * (math.cos(math.pi * i / _T) + 1) / 2
            for i in range(_T + 1)
        )
        self.best_model = None
        
        
        #forcasting
        self.Decoder_patches = LinearDecoder(emb_dim=config["predictor_embed_dim"], patch_size=config["patch_size"])
        self.forcast_train = forcasting_train
        self.forcast_val = forcasting_val
        self.forcast_test = forcasting_test
        self.epoch_t = config["epoch_t"]
        self.Context_t = config["context_t"]
        self.Patches_to_forcast = config["patches_to_forcast"]
        self.patches_size_forecasting = config["patches_size_forecasting"]

        self.encoder_for = Encoder(
            num_patches=config["ratio_patches"],
            dim_in=input_dim,
            embed_dim=config["encoder_embed_dim"],
            nhead=config["nhead"],
            num_layers=config["num_encoder_layers"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"],
            attn_drop_rate=config["attn_drop_rate"],
            pe='sincos',
            learn_pe=False,
            res_attention=True,
        )
        self.predictor_for = Predictor(
            num_patches=config["ratio_patches"],
            embed_dim=config["encoder_embed_dim"],
            nhead=config["predictor_nhead"],
            num_layers=config["predictor_num_layers"],
            config=config,
        )

# Bind loss helpers from losses.py as methods on the class
JEPA._calculate_vicreg_loss       = _calculate_vicreg_loss

# Bind forecasting methods from Forecasting.py as methods on the class
JEPA.forcasting_zeroshot      = forcasting_zeroshot

# Bind classification methods from Classification.py as methods on the class
JEPA.classification_zeroshot  = classification_zeroshot

# Bind anomaly detection methods from Anomaly.py as methods on the class
JEPA.anomaly_zeroshot         = anomaly_zeroshot

# Bind training methods from Training.py as methods on the class
JEPA.compute_jepa_loss           = compute_jepa_loss
JEPA.evaluate                    = evaluate
JEPA.save_model                  = save_model
JEPA.train_and_evaluate          = train_and_evaluate
