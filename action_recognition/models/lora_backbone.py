"""
VideoMAEv2 백본(K710 finetune) + LoRA(qkv) + Linear head 통합 모델.

학습 대상: LoRA params + head
Freeze:    백본 원본 가중치

CRITICAL: 이 모델의 Attention.forward는 F.linear(x, self.qkv.weight, bias)를 직접 호출 →
          PEFT LoRA wrapper(self.qkv 모듈)를 우회. 그래서 _patched_attention_forward 로
          monkey-patch해서 self.qkv(x) 형태로 호출 강제 → LoRA delta 정상 적용.
"""
import types

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

from models.head import VideoMAEv2Head


def _patched_attention_forward(self, x):
    """Attention.forward replacement that routes through self.qkv(x).

    Original (modeling_videomaev2.py:184):
        qkv = F.linear(x, self.qkv.weight, bias=qkv_bias)   # ← LoRA bypassed
    New:
        qkv = self.qkv(x) + qkv_bias                         # ← LoRA delta applied
    """
    B, N, _ = x.shape
    qkv = self.qkv(x)                                        # base + LoRA delta (PEFT-wrapped)
    if self.q_bias is not None:
        qkv_bias = torch.cat(
            (self.q_bias, torch.zeros_like(self.v_bias), self.v_bias)
        )
        qkv = qkv + qkv_bias
    qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


PRETRAIN_DIR = "/root/workspace/Drowning-Detection-VLM/weights/VideoMAEv2-Base-pretrain"
FINETUNE_PTH = "/root/workspace/Drowning-Detection-VLM/weights/VideoMAEv2-K710-finetune/distill/vit_b_k710_dl_from_giant.pth"
HEAD_CKPT_DEFAULT = "/root/workspace/Drowning-Detection-VLM/action_recognition/checkpoints/head_finetune_best64.pt"


class VideoMAEv2_LoRA_Linear(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        head_ckpt_path: str = HEAD_CKPT_DEFAULT,
    ):
        super().__init__()

        # 백본 구조 생성
        backbone = AutoModel.from_pretrained(PRETRAIN_DIR, trust_remote_code=True)

        # K710 finetune weight 로드
        # head.* 는 K710 분류기 head이므로 제외
        ckpt = torch.load(FINETUNE_PTH, map_location="cpu", weights_only=False)
        ft_sd = ckpt["module"]
        new_sd = {f"model.{k}": v for k, v in ft_sd.items() if not k.startswith("head.")}
        missing, unexpected = backbone.load_state_dict(new_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Load mismatch: missing={missing}, unexpected={unexpected}")

        # Gradient checkpointing 비활성.
        # 이 모델은 cp.checkpoint(blk, x)를 reentrant 기본 모드로 사용 → LoRA gradient 손실 확인됨.
        # bf16 autocast + batch=2면 16GB VRAM에 들어가서 굳이 필요 없음.
        # 메모리 부족 시 backbone.model.with_cp=True + reentrant=False 패치로 다시 활성 가능.
        backbone.model.with_cp = False

        # LoRA 부착 (get_peft_model이 base 자동 freeze + qkv에 LoRA 삽입)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["qkv"],
            bias="none",
            task_type=None,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)

        # CRITICAL: 모든 Attention.forward를 patched 버전으로 교체
        # (원본 forward가 F.linear(x, self.qkv.weight)로 LoRA wrapper 우회하는 문제 fix)
        n_patched = 0
        for _, mod in self.backbone.named_modules():
            if type(mod).__name__ == "Attention":
                mod.forward = types.MethodType(_patched_attention_forward, mod)
                n_patched += 1
        print(f"[patch] Patched {n_patched} Attention.forward (LoRA bypass fix)")

        # Head 생성 + 기존 best 가중치 로드
        self.head = VideoMAEv2Head(in_dim=768, num_classes=num_classes)
        if head_ckpt_path is not None:
            ckpt_head = torch.load(head_ckpt_path, map_location="cpu", weights_only=False)
            self.head.load_state_dict(ckpt_head["state_dict"])

        # Trainable 검증
        self.backbone.print_trainable_parameters()

    def forward(self, x):
        feat = self.backbone(x)          # (B, 768) — 백본 내부 mean pooling 포함
        logits = self.head(feat)         # (B, num_classes)
        return logits

    # ---------- 저장/로드 헬퍼 ----------
    def trainable_state_dict(self):
        """LoRA + head 가중치만 추출 (백본 원본은 .pth에서 다시 로드 가능)."""
        sd = {}
        for k, v in self.backbone.state_dict().items():
            if "lora_" in k:
                sd[f"backbone.{k}"] = v
        for k, v in self.head.state_dict().items():
            sd[f"head.{k}"] = v
        return sd

    def load_trainable_state_dict(self, state):
        """trainable_state_dict() 결과를 다시 로드."""
        backbone_sd = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        head_sd = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
        self.backbone.load_state_dict(backbone_sd, strict=False)
        self.head.load_state_dict(head_sd)


if __name__ == "__main__":
    net = VideoMAEv2_LoRA_Linear()
    print(net)
    x = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        out = net(x)
    print("output shape:", out.shape)  # 예상: torch.Size([1, 2])
