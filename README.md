## Prerequisites

- GPU with ≥16 GB VRAM
- System RAM ≥32 GB
- ~30 GB free disk space
- CUDA drivers installed (`nvidia-smi` should work)

---

## 1. Create a virtual environment

```bash
python3 -m venv openvla-env
source openvla-env/bin/activate
```

> If you need Python 3.10 specifically (e.g. to use flash-attn later):
> ```bash
> sudo apt install python3.10 python3.10-venv
> python3.10 -m venv openvla-env
> source openvla-env/bin/activate
> ```

---

## 2. Install PyTorch

**Important:** Standard PyTorch releases only support up to sm_90 (Hopper). Blackwell (sm_120) requires a nightly build with CUDA 13.0+.

```bash
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu130
```

Verify your GPU is working:

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
x = torch.randn(3,3).cuda()
print('Kernel test passed:', (x @ x).shape)
"
```

---

## 3. Install dependencies

```bash
pip install "transformers==4.40.1" \
            "timm>=0.9.10,<1.0.0" \
            pillow \
            accelerate \
            huggingface_hub
```

> **Note:** Pin both packages to these versions exactly:
> - `transformers==4.40.1` — newer versions break OpenVLA's custom model class
> - `timm>=0.9.10,<1.0.0` — OpenVLA hardcodes this range; timm 1.x will throw a `NotImplementedError`

---

## 4. Download the model

```bash
pip install "huggingface_hub[cli]"

huggingface-cli download openvla/openvla-7b \
  --local-dir ./models/openvla-7b
```

This downloads ~15 GB. Verify it completed:

```bash
ls ./models/openvla-7b/
# Should include: config.json, model.safetensors.index.json, tokenizer files
```

> **Note:** Do not use `attn_implementation="flash_attention_2"` — it should be built properly we are skipping it for now.


## 5. Download the images

```
wget -r -np -nd -A "*.jpg,*.png"   "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/"   -P frames/
```

## 6. Run it:

```bash
python3 vla_inference_1.py "frames/*.jpg" --instruction "open the drawer"
```

Expected output:
```
Action: [-0.024  0.008 -0.003  0.082  0.077  0.032  0.996] ... (as many as there are images)
```

---

---

## Troubleshooting

| Error | Fix |
|---|---|
| `FlashAttention2 cannot be used` | Remove `attn_implementation="flash_attention_2"` from `from_pretrained()` |
| `TIMM Version must be >= 0.9.10 and < 1.0.0` | `pip install "timm>=0.9.10,<1.0.0"` |
| `no kernel image is available for execution on the device` | You're on the wrong PyTorch build — reinstall with `--index-url .../nightly/cu130` |
| PyTorch still shows old version after reinstall | Check `which pip` — make sure it points to your venv, not the system Python |
| OOM errors | Add `load_in_4bit=True` to `from_pretrained()` and `pip install bitsandbytes` |

---

## Next steps

- **Fine-tune on your own robot data** — use the LoRA script in `vla-scripts/finetune.py`
- **Connect to a live camera** — swap `Image.open()` for a webcam or ROS topic
- **OpenVLA-OFT** — the official follow-up with 25–50× faster inference via parallel decoding: [openvla-oft.github.io](https://openvla-oft.github.io)
- **Full docs** — [github.com/openvla/openvla](https://github.com/openvla/openvla)
