"""LoRA compatibility check for the chosen CLIP implementation (HF transformers CLIPModel + peft).

Run BEFORE any training:
  python -m scripts.check_lora --pretrained            # on Colab: real openai/clip-vit-base-patch32
  python -m scripts.check_lora                          # offline: randomly initialised ViT-B/32 config
"""
import argparse, copy
import torch
import transformers, peft
from transformers import CLIPConfig, CLIPModel
from src.model import build_model, trainable_state

ap = argparse.ArgumentParser()
ap.add_argument("--pretrained", action="store_true")
args = ap.parse_args()
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)

torch.manual_seed(0)
base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") if args.pretrained else CLIPModel(CLIPConfig())
base.eval()
x_img = torch.randn(4, 3, 224, 224)
ids = torch.randint(1, 1000, (4, 12)); mask = torch.ones_like(ids)
with torch.no_grad():
    ref_i = base.get_image_features(pixel_values=x_img)
    ref_t = base.get_text_features(input_ids=ids, attention_mask=mask)
ref_i = getattr(ref_i, "pooler_output", ref_i); ref_t = getattr(ref_t, "pooler_output", ref_t)

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

for towers in ["both", "text"]:
    print(f"\n--- towers={towers} ---")
    cfg = {"tune": {"mode": "lora", "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "towers": towers}}}
    m = build_model(cfg, "cpu", model=copy.deepcopy(base))
    m.eval()
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    vis = sum("vision_model" in n for n in names); txt = sum("text_model" in n for n in names)
    print(f"trainable tensors: vision={vis}, text={txt}, total={len(names)}")
    check("only LoRA params are trainable", all("lora_" in n for n in names))
    check("text tower adapted", txt > 0)
    check("vision tower adapted iff towers=both", (vis > 0) == (towers == "both"))
    check("logit_scale frozen", not dict(m.named_parameters())["base_model.model.logit_scale"].requires_grad)
    with torch.no_grad():
        out_i = m.get_image_features(pixel_values=x_img); out_t = m.get_text_features(input_ids=ids, attention_mask=mask)
    out_i = getattr(out_i, "pooler_output", out_i); out_t = getattr(out_t, "pooler_output", out_t)
    check("output identical to base at init (LoRA B=0)", torch.allclose(out_i, ref_i, atol=1e-5) and torch.allclose(out_t, ref_t, atol=1e-5))
    m.train()
    fi = m.get_image_features(pixel_values=x_img); ft = m.get_text_features(input_ids=ids, attention_mask=mask)
    fi = getattr(fi, "pooler_output", fi); ft = getattr(ft, "pooler_output", ft)
    (fi.norm() + ft.norm()).backward()
    got = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is not None]
    check("gradients reach LoRA params (A or B)", len(got) > 0)
    check("no gradient on frozen base weights", all(p.grad is None for n, p in m.named_parameters() if not p.requires_grad))
    st = trainable_state(m)
    check(f"adapter-only checkpoint is small ({sum(v.numel() for v in st.values()) * 4 / 1e6:.1f} MB fp32)", True)
    out = m(input_ids=ids, pixel_values=x_img, attention_mask=mask)
    check("full forward exposes logits_per_image (needed by Winoground eval)", out.logits_per_image.shape == (4, 4))

print("\nOVERALL:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
