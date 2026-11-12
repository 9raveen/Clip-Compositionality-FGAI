"""Model construction: pretrained CLIP + optional LoRA / full fine-tuning."""
import torch
from transformers import CLIPModel


def lora_target_modules(towers: str):
    """q/v attention projections. 'both' matches vision+text towers, 'text' only the text tower."""
    if towers == "both":
        return ["q_proj", "v_proj"]
    if towers == "text":
        return r".*text_model.*\.(q_proj|v_proj)"
    if towers == "vision":
        return r".*vision_model.*\.(q_proj|v_proj)"
    raise ValueError(towers)


def build_model(cfg: dict, device, model=None):
    """cfg['tune']['mode'] in {'none', 'lora', 'full'}. Returns the (possibly PEFT-wrapped) model."""
    model = model if model is not None else CLIPModel.from_pretrained(cfg["model"])
    mode = cfg.get("tune", {}).get("mode", "none")
    for p in model.parameters():
        p.requires_grad = False
    if mode == "full":
        for n, p in model.named_parameters():
            p.requires_grad = (n != "logit_scale")      # keep CLIP's learned temperature fixed
    elif mode == "lora":
        from peft import LoraConfig, get_peft_model
        l = cfg["tune"]["lora"]
        lcfg = LoraConfig(r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"],
                          target_modules=lora_target_modules(l.get("towers", "both")), bias="none")
        model = get_peft_model(model, lcfg)             # freezes base weights, adds trainable adapters
    elif mode != "none":
        raise ValueError(mode)
    model.to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] mode={mode} trainable={n_tr:,} / {n_all:,} ({100 * n_tr / n_all:.2f}%)")
    return model


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def feat(x):
    """get_*_features returns a tensor in transformers 4.x and a ModelOutput in 5.x."""
    return getattr(x, "pooler_output", x)
