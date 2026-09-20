# Laya Azure Deployment & Production Guidance

Follow-up note for scaling and deploying the self-hosted Laya System 1 decision engine to Azure.

---

## 1. Recommended Azure VM SKUs & Sizing

Because Laya uses a bidirectional transformer encoder backbone (**ModernBERT-large**, 421M parameters) without autoregressive token generation, its VRAM and compute requirements are compact:

### Workhorse: `Standard_NC4as_T4_v3` (Parity with Research Paper)
- **Hardware**: 1× NVIDIA Tesla T4 (16 GB VRAM), 4 vCPUs, 28 GB RAM.
- **Latency Target**:
  - Single question: **~30–35 ms**
  - Full article (15–25 questions in 1 forward pass): **~75–150 ms**
- **Estimated Cost**:
  - Pay-as-you-go: **~$0.526 / hour** (~$12.60 / day, ~$380 / month 24/7)
  - 1-Year Reserved: **~$0.33 / hour** (~$240 / month)
  - Spot Instance: **~$0.16 / hour** (~$3.80 / day for batch testing)

### High Throughput: `Standard_NV6ads_A10_v5` (Fastest / Production Scale)
- **Hardware**: 1× NVIDIA Ampere A10 (24 GB VRAM), 6 vCPUs, 55 GB RAM.
- **Latency Target**:
  - Full article (15–25 questions): **~25–45 ms** (significantly faster than cloud TypeSafe Jev)
- **Estimated Cost**:
  - Pay-as-you-go: **~$0.75 – $0.90 / hour**
  - Spot Instance: **~$0.25 / hour**

### Periodic / Serverless: Azure Container Apps (ACA) with GPU
- Scale-to-zero when no benchmarks are running; spins up replicas on demand during batch processing.

---

## 2. Co-location & Security Benefits
- **VNet Proximity**: Deploying Laya in the same Azure region (e.g. `East US`) as the MySQL database and Blob Storage container eliminates WAN egress and reduces article fetch round-trips from ~200ms to **1–3 ms**.
- **Data Perimeter**: Articles, broadcast transcripts, and proprietary curation prompts remain strictly within the Azure private network without sending payloads to external SaaS providers.

---

## 3. Containerization Spec
- **Base Image**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`
- **Inference Server**: FastAPI + Uvicorn with `torch.compile(model, mode="reduce-overhead")` and FP16 autocast (`torch.cuda.amp.autocast`).
- **Endpoint Contract**: Compatible with the existing `/api/alpha/decisions` contract used by `typesafe_runner.py`.
