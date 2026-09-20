import json
import logging
import os
import random
import time
from typing import Any, Optional

import laya
import torch
from laya.common import collate_items
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_laya")


def evaluate_model(
    model: torch.nn.Module,
    val_items: list[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
    batch_size: int = 16,
    max_eval_items: int = 120,
) -> dict[str, Any]:
    """Evaluates decision accuracy on validation sequences across all primitives."""
    model.eval()
    eval_set = val_items[:max_eval_items]
    correct_by_kind: dict[str, int] = {}
    total_by_kind: dict[str, int] = {}
    total_loss = 0.0
    n_batches = 0

    val_logits_by_type: dict[int, list[tuple[list[float], list[float]]]] = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for i in range(0, len(eval_set), batch_size):
            chunk = eval_set[i : i + batch_size]
            b = collate_items([chunk], pad_token_id)
            if b is None:
                continue

            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            marker_pos = b["marker_pos"].to(device)
            marker_mask = b["marker_mask"].to(device)
            qtype = b["qtype"].to(device)
            target = b["target"].to(device)
            labels = b["label"].to(device)

            logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
            masked_logits = logits.masked_fill(~marker_mask, -1e4)
            ce = -(target * torch.log_softmax(masked_logits, -1)).sum(-1).mean()
            total_loss += ce.item()
            n_batches += 1

            preds = masked_logits.argmax(-1)
            matches = (preds == labels).cpu().numpy()

            for it, match, qt, logit_row, targ_row, mmask in zip(
                chunk, matches, qtype.cpu().numpy(), logits.cpu().numpy(), target.cpu().numpy(), marker_mask.cpu().numpy()
            ):
                kind = it.get("decision_kind", "unknown")
                correct_by_kind[kind] = correct_by_kind.get(kind, 0) + int(match)
                total_by_kind[kind] = total_by_kind.get(kind, 0) + 1

                k = int(mmask.sum())
                val_logits_by_type[int(qt)].append((logit_row[:k].tolist(), targ_row[:k].tolist()))

    overall_correct = sum(correct_by_kind.values())
    overall_total = max(1, sum(total_by_kind.values()))
    accuracy_by_kind = {
        kind: round((correct_by_kind.get(kind, 0) / max(1, total_by_kind[kind])) * 100.0, 2)
        for kind in total_by_kind
    }

    return {
        "overall_accuracy_pct": round((overall_correct / overall_total) * 100.0, 2),
        "overall_loss": round(total_loss / max(1, n_batches), 4),
        "by_kind": accuracy_by_kind,
        "counts": {kind: f"{correct_by_kind.get(kind, 0)}/{total_by_kind[kind]}" for kind in total_by_kind},
        "raw_logits_by_type": val_logits_by_type,
    }


def fit_temperature_scaling(val_data_by_type: dict[int, list[tuple[list[float], list[float]]]]) -> list[float]:
    """Fits scalar temperature per question type (choice=0, score=1, noul=2) using LBFGS."""
    temperatures = [1.0, 1.0, 1.0]

    for qt in (0, 1, 2):
        samples = val_data_by_type.get(qt, [])
        if len(samples) < 15:
            continue

        kmax = max(len(z) for z, _ in samples)
        z_tensor = torch.full((len(samples), kmax), -1e4)
        t_tensor = torch.zeros((len(samples), kmax))

        for i, (z, t) in enumerate(samples):
            z_tensor[i, : len(z)] = torch.tensor(z)
            t_tensor[i, : len(t)] = torch.tensor(t, dtype=torch.float32)

        log_temp = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temp], lr=0.08, max_iter=80)

        def closure(opt=optimizer, lt=log_temp, zt=z_tensor, tt=t_tensor):
            opt.zero_grad()
            t_val = lt.exp()
            scaled = zt / t_val
            loss = -(tt * torch.log_softmax(scaled, -1)).sum(-1).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        final_t = float(torch.clamp(log_temp.exp(), 0.1, 8.0).item())
        temperatures[qt] = round(final_t, 3)
        logger.info(f"Fitted temperature for qtype {qt}: {temperatures[qt]}")

    return temperatures


def train_laya(
    dataset_dir: str = "../runs/laya_training_data_10k",
    output_dir: str = "../runs/laya_finetuned_10k",
    epochs: int = 1,
    micro_batch: int = 16,
    grad_accum: int = 2,
    lr: float = 5.0e-5,
    max_train_samples: Optional[int] = None,
    max_val_samples: int = 300,
    top_layers: int = 6,
) -> dict[str, Any]:
    """Fine-tunes Laya on Curation Engine decision sequences using CUDA / MPS / CPU."""
    if torch.cuda.is_available():
        device_name = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    logger.info(f"Using compute device: {device_name}")

    train_path = os.path.join(dataset_dir, "train_items.pt")
    val_path = os.path.join(dataset_dir, "val_items.pt")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Training dataset not found in {dataset_dir}. Run dataset_builder.py first.")

    train_items: list[dict[str, Any]] = torch.load(train_path, weights_only=False)
    val_items: list[dict[str, Any]] = torch.load(val_path, weights_only=False)

    if max_train_samples and len(train_items) > max_train_samples:
        random.seed(42)
        random.shuffle(train_items)
        train_items = train_items[:max_train_samples]

    logger.info(f"Loaded dataset: {len(train_items)} train sequences, {len(val_items)} validation sequences")

    # Load base pretrained model
    logger.info("Loading pretrained Laya model 'convaiinnovations/laya:typed-decisions'...")
    base_agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions", device=device_name)
    model = base_agent.model
    tok = base_agent.tok
    cfg = dict(base_agent.cfg)

    # Initial pre-training evaluation
    logger.info("Evaluating base model on validation split before fine-tuning...")
    pre_eval = evaluate_model(model, val_items, tok.pad_token_id, device, max_eval_items=max_val_samples)
    logger.info(f"Pre-training validation accuracy: {pre_eval['overall_accuracy_pct']}% | {pre_eval['by_kind']}")

    # Freeze bottom layers, train top N layers + decision head
    for param in model.encoder.parameters():
        param.requires_grad = False

    encoder_layers = getattr(model.encoder, "layers", None)
    if isinstance(encoder_layers, (torch.nn.ModuleList, list)):
        for layer in list(encoder_layers)[-top_layers:]:
            if isinstance(layer, torch.nn.Module):
                for param in layer.parameters():
                    param.requires_grad = True
    # Decision head parameters always trainable
    if model.head is not None:
        for param in model.head.parameters():
            param.requires_grad = True
    for param in model.type_emb.parameters():
        param.requires_grad = True
    for param in model.scorer.parameters():
        param.requires_grad = True
    for param in model.act_head.parameters():
        param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_count:,} / {total_count:,} ({trainable_count / total_count * 100:.1f}%)")

    model.train()
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    use_autocast = (device_name == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_autocast)

    total_steps = (len(train_items) // (micro_batch * grad_accum)) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6)

    t_start = time.perf_counter()
    logger.info(f"Starting fine-tuning: {epochs} epochs | ~{total_steps} optimizer steps | grad_accum={grad_accum} | autocast={use_autocast}")

    step_count = 0
    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()
        random.seed(42 + epoch)
        random.shuffle(train_items)

        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        running_loss = 0.0
        n_batches = 0

        for b_idx in range(0, len(train_items), micro_batch):
            chunk = train_items[b_idx : b_idx + micro_batch]
            if not chunk:
                continue

            b = collate_items([chunk], tok.pad_token_id)
            if b is None:
                continue

            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            marker_pos = b["marker_pos"].to(device)
            marker_mask = b["marker_mask"].to(device)
            qtype = b["qtype"].to(device)
            target = b["target"].to(device)

            with torch.autocast(device_type=device_name, enabled=use_autocast):
                logits, act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
                loss_ce = -(target * torch.log_softmax(logits.masked_fill(~marker_mask, -1e4), -1)).sum(-1).mean()
                loss = loss_ce / grad_accum + 0.0 * act.sum()

            if use_autocast:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_step += 1
            running_loss += loss_ce.detach()
            n_batches += 1

            if accum_step % grad_accum == 0 or (b_idx + micro_batch) >= len(train_items):
                if use_autocast:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
                if step_count % 50 == 0:
                    logger.info(f"Step {step_count}/{total_steps} | Batch Loss: {loss_ce.item():.4f}")

        epoch_duration = time.perf_counter() - epoch_t0
        avg_epoch_loss = float(running_loss.item()) / max(1, n_batches)
        logger.info(
            f"Epoch {epoch + 1}/{epochs} complete in {epoch_duration:.1f}s | "
            f"Avg Loss: {avg_epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    total_duration = time.perf_counter() - t_start
    logger.info(f"Fine-tuning complete in {total_duration:.1f} seconds ({total_duration / 60.0:.2f} min)!")

    # Post-training validation
    logger.info("Evaluating fine-tuned model on validation split...")
    post_eval = evaluate_model(model, val_items, tok.pad_token_id, device, max_eval_items=max_val_samples)
    logger.info(f"Post-training validation accuracy: {post_eval['overall_accuracy_pct']}% | {post_eval['by_kind']}")

    # Fit temperature calibration
    logger.info("Fitting temperature calibration on validation logits...")
    calibrated_temps = fit_temperature_scaling(post_eval["raw_logits_by_type"])
    model.temperature.data = torch.tensor(calibrated_temps, dtype=torch.float32)

    # Save fine-tuned artifacts
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "model.safetensors")
    save_file(model.state_dict(), weights_path)

    cfg["temperature"] = calibrated_temps
    cfg["finetuned_on"] = "curation_engine_audit_blobs"
    cfg["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(output_dir, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Save tokenizer
    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    tok.save_pretrained(tok_dir)

    # Save encoder config
    enc_dir = os.path.join(output_dir, "encoder")
    os.makedirs(enc_dir, exist_ok=True)
    model.encoder.config.save_pretrained(enc_dir)

    summary = {
        "device": device_name,
        "epochs": epochs,
        "train_samples": len(train_items),
        "val_samples": min(len(val_items), max_val_samples),
        "total_training_duration_s": round(total_duration, 1),
        "pre_training_eval": {
            "overall_accuracy_pct": pre_eval["overall_accuracy_pct"],
            "by_kind": pre_eval["by_kind"],
            "counts": pre_eval["counts"],
        },
        "post_training_eval": {
            "overall_accuracy_pct": post_eval["overall_accuracy_pct"],
            "by_kind": post_eval["by_kind"],
            "counts": post_eval["counts"],
        },
        "calibrated_temperatures": calibrated_temps,
        "checkpoint_dir": os.path.abspath(output_dir),
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved fine-tuned checkpoint and summary to {output_dir}")
    return summary

if __name__ == "__main__":
    train_laya(epochs=1, max_train_samples=None, max_val_samples=300)
