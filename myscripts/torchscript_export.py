import torch
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model

# === EDIT THESE ===
CFG = r"D:/Deep_Learning/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CKPT = r"D:/Deep_Learning/Grounding-Dino-FineTuning/weights_myds/epoch_100.pth"
OUT  = r"D:/HybridDetector/onnx/grounding_dino_scripted.pt"  # TorchScript output
# ===================

# 1) Build model and load weights
args = SLConfig.fromfile(CFG)
model = build_model(args)
ckpt = torch.load(CKPT, map_location="cpu")
state = ckpt.get("model_state_dict", ckpt)
state = {k.replace("module.", ""): v for k,v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
model.eval()

# 2) Wrap forward to a stable signature:
# inputs: images[N,3,H,W] float32 (normalized), input_ids[1,L] int64, attention_mask[1,L] int64
# outputs: pred_logits[1,Q,C], pred_boxes[1,Q,4] (cx,cy,w,h normalized)
class GDinoWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, images, input_ids, attention_mask):
        # GroundingDINO expects dicts; we pass tokenized text via kwargs
        # Your repo’s model returns a dict with 'pred_logits' and 'pred_boxes'
        out = self.m(images, input_ids=input_ids, attention_mask=attention_mask)
        return out["pred_logits"], out["pred_boxes"]

wrapper = GDinoWrapper(model)

# 3) Trace/script
# Use scripting for control flow robustness
scripted = torch.jit.script(wrapper)

# 4) Save TorchScript
scripted.save(OUT)
print("Saved TorchScript:", OUT)
