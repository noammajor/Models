import copy
import math

import torch
import torch.nn as nn

from Discrete_JEPA.Encoder import Encoder
from Discrete_JEPA.Predictors import DiscreteJEPAPredictor as Predictor
from Discrete_JEPA.Decoder import LinearDecoder
from Discrete_JEPA.VQ import *
from Discrete_JEPA.losses import (
    _calculate_vicreg_loss,
    _calculate_token_diversity_loss,
    _grounding_loss,
    _cross_decorrelation_loss,
)
from Discrete_JEPA.Forecasting import (
    predictor_forecasting,
    finetuning_forecasting,
    forcasting_zeroshot,
    predictor_s2p_p2p_forecasting,
)
from Discrete_JEPA.Training import (
    _compute_global_stats,
    compute_discrete_jepa_loss,
    evaluate,
    save_model,
    train_and_evaluate,
)

class DiscreteJEPA(nn.Module):
    def __init__(self, config, input_dim, num_patches, steps_per_epoch, train_loader, val_loader, test_loader, forcasting_train, forcasting_val, forcasting_test):
        super(DiscreteJEPA, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.encoder = Encoder(
            num_patches=config["ratio_patches"],
            num_semantic_tokens=config["num_semantic_tokens"],
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
            num_semantic_tokens=config["num_semantic_tokens"],
            embed_dim=config["encoder_embed_dim"],
            nhead=config["predictor_nhead"],
            num_layers=config["predictor_num_layers"],
            config=config,
        )
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=config["codebook_size"],
            embedding_dim=config["encoder_embed_dim"],
            commitment_cost=config["commitment_cost"]
        )

        encoder_params = list(self.encoder.parameters())
        other_params_pred = list(self.predictor.parameters())
        codebook_params = list(self.vector_quantizer.parameters())

        # Switched to AdamW for better transformer training stability
        self.optimizer = torch.optim.SGD([
            {
                "params": encoder_params,
                "lr": config["lr"],
                "weight_decay": config["weight_decay"],
                "momentum": 0.9  # Typical starting value for SGD
            },
            {
                "params": other_params_pred,
                "lr": config["lr_pred"],
                "weight_decay": config["weight_decay_pred"],
                "momentum": 0.9
            },
            {
                "params": codebook_params,
                "lr": config["codebook_lr"],
                "weight_decay": 0.0,  # no decay: codebook vectors are discrete symbols
                "momentum": 0.9
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
            max_lr=[config["lr"], config["lr_pred"], config["codebook_lr"]],
            epochs=config["num_epochs"],
            steps_per_epoch=self.steps_per_epoch,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1000.0
            )
        else:
            self.scheduler = None
        
        self.encoder_ema = copy.deepcopy(self.encoder)
        self.encoder_ema.jepa = True
        self.encoder_ema.type_enc = "target"
        self.encoder_ema.eval()  # Target encoder always deterministic — no dropout
        self.codebooksize = config["codebook_size"]
        self.checkpoint_save = self.config["checkpoint_save"]
        self.checkpoint_print = self.config["checkpoint_print"]
        self.path_save = self.config["path_save"]
        self.clip_grad = self.config["clip_grad"]
        self.warmup = self.config["warmup_ratio"] * self.config["num_epochs"]
        _T = int(self.config["num_epochs"] * len(train_loader))
        _m0 = self.config["ema_momentum"]
        self.ema_scheduler = (
            1 - (1 - _m0) * (math.cos(math.pi * i / _T) + 1) / 2
            for i in range(_T + 1)
        )


        self.best_model = None
        self.log_file = "perplexity_log.csv"
        self.logsss ="info.csv"
        with open(self.log_file, "w") as f:
            f.write("epoch,step,type,perplexity\n")
        with open(self.logsss, "w") as g:
            g.write("epoch,step,type,perplexity\n")
        
        
        #forcasting
        self.Decoder_patches = LinearDecoder(emb_dim=config["predictor_embed_dim"], patch_size=config["patch_size"])
        self.Decoder_semantic = LinearDecoder(emb_dim=config["predictor_embed_dim"], patch_size=config["patch_size"])
        self.forcast_train = forcasting_train
        self.forcast_val = forcasting_val
        self.forcast_test = forcasting_test
        self.epoch_t = config["epoch_t"]
        self.Context_t = config["context_t"]
        self.Patches_to_forcast = config["patches_to_forcast"]
        self.patches_size_forecasting = config["patches_size_forecasting"]

        self.encoder_for = Encoder(
            num_patches=len(self.train_loader.dataset[0][0]),
            num_semantic_tokens=config["num_semantic_tokens"],
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
            num_patches=len(self.train_loader.dataset[0][0]),
            num_semantic_tokens=config["num_semantic_tokens"],
            embed_dim=config["encoder_embed_dim"],
            nhead=config["predictor_nhead"],
            num_layers=config["predictor_num_layers"],
            config=config,
        )
        self._compute_global_stats()



# Bind loss helpers from losses.py as methods on the class
DiscreteJEPA._calculate_vicreg_loss       = _calculate_vicreg_loss
DiscreteJEPA._calculate_token_diversity_loss = _calculate_token_diversity_loss
DiscreteJEPA._grounding_loss              = _grounding_loss
DiscreteJEPA._cross_decorrelation_loss    = _cross_decorrelation_loss

# Bind forecasting methods from Forecasting.py as methods on the class
DiscreteJEPA.predictor_forecasting          = predictor_forecasting
DiscreteJEPA.finetuning_forecasting         = finetuning_forecasting
DiscreteJEPA.forcasting_zeroshot            = forcasting_zeroshot
DiscreteJEPA.predictor_s2p_p2p_forecasting  = predictor_s2p_p2p_forecasting

# Bind training methods from Training.py as methods on the class
DiscreteJEPA._compute_global_stats       = _compute_global_stats
DiscreteJEPA.compute_discrete_jepa_loss  = compute_discrete_jepa_loss
DiscreteJEPA.evaluate                    = evaluate
DiscreteJEPA.save_model                  = save_model
DiscreteJEPA.train_and_evaluate          = train_and_evaluate
