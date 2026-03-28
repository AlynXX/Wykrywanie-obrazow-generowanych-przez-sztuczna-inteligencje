from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import create_model

AUTO_BATCH_ALGO_VERSION = 1


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    keywords = (
        "out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cuda out of memory",
    )
    return any(keyword in message for keyword in keywords)


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()


def _is_compile_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized not in {"", "0", "false", "none", "off", "no"}


def resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Nieobslugiwany amp_dtype: {dtype_name}")


def configure_compile_caches(output_dir: Path):
    cache_root = output_dir / ".torch_compile_cache"
    triton_cache_dir = cache_root / "triton"
    inductor_cache_dir = cache_root / "inductor"
    temp_dir = cache_root / "tmp"
    triton_cache_dir.mkdir(parents=True, exist_ok=True)
    inductor_cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache_dir.resolve()))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache_dir.resolve()))
    os.environ.setdefault("TMP", str(temp_dir.resolve()))
    os.environ.setdefault("TEMP", str(temp_dir.resolve()))
    os.environ.setdefault("TMPDIR", str(temp_dir.resolve()))
    tempfile.tempdir = str(temp_dir.resolve())


def _cache_enabled_compiled(auto_batch_cfg: dict[str, Any], compile_value: Any) -> bool:
    if not bool(auto_batch_cfg.get("cache_enabled", True)):
        return False
    return _is_compile_enabled(compile_value)


def _resolve_cache_path(auto_batch_cfg: dict[str, Any]) -> Path:
    cache_path_value = auto_batch_cfg.get("cache_path", "models/auto_batch_compiled_cache.json")
    cache_path = Path(cache_path_value)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return cache_path


def _load_cache(cache_path: Path) -> dict[str, Any]:
    default_cache = {"version": 1, "entries": {}}
    if not cache_path.exists():
        return default_cache
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return default_cache
    if not isinstance(payload, dict):
        return default_cache
    payload.setdefault("version", 1)
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        payload["entries"] = {}
    return payload


def _save_cache(cache_path: Path, cache_payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")


def _resolve_torch_device(device: Any) -> torch.device:
    if isinstance(device, int):
        if device >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")

    if isinstance(device, str):
        normalized = device.strip().lower()
        if normalized in {"", "auto"}:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if normalized.isdigit() and torch.cuda.is_available():
            return torch.device(f"cuda:{normalized}")
        try:
            return torch.device(device)
        except Exception:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _device_fingerprint(device: Any) -> dict[str, Any]:
    torch_device = _resolve_torch_device(device)
    fingerprint = {"requested_device": str(device), "resolved_device": str(torch_device)}
    if torch_device.type == "cuda" and torch.cuda.is_available():
        index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        fingerprint.update(
            {
                "name": str(props.name),
                "index": int(index),
                "total_memory": int(getattr(props, "total_memory", 0)),
                "cc": f"{int(props.major)}.{int(props.minor)}",
            }
        )
    return fingerprint


def _batch_cache_key(
    *,
    model_name: str,
    num_classes: int,
    image_size: int,
    device: Any,
    compile_value: Any,
    auto_batch_cfg: dict[str, Any],
    min_batch: int,
    max_batch: int,
    multiple_of: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[str, dict[str, Any]]:
    key_payload = {
        "algo_version": AUTO_BATCH_ALGO_VERSION,
        "model_name": model_name,
        "num_classes": int(num_classes),
        "imgsz": int(image_size),
        "device": _device_fingerprint(device),
        "compile_enabled": _is_compile_enabled(compile_value),
        "compile_mode": str(auto_batch_cfg.get("compile_mode", "default")),
        "compile_backend": str(auto_batch_cfg.get("compile_backend", "inductor")),
        "compile_dynamic": bool(auto_batch_cfg.get("compile_dynamic", False)),
        "probe_amp": bool(amp_enabled),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "max_vram_utilization": float(auto_batch_cfg.get("max_vram_utilization", 0.9)),
        "max_vram_metric": str(auto_batch_cfg.get("max_vram_metric", "allocated")).strip().lower(),
        "synthetic_warmup_steps": int(auto_batch_cfg.get("synthetic_warmup_steps", 1)),
        "synthetic_measure_steps": int(auto_batch_cfg.get("synthetic_measure_steps", 1)),
        "min_batch": int(min_batch),
        "max_batch": int(max_batch),
        "multiple_of": int(multiple_of),
    }
    key_text = json.dumps(key_payload, sort_keys=True, ensure_ascii=True, default=str)
    cache_key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    return cache_key, key_payload


def _read_cached_batch(
    *,
    cache_payload: dict[str, Any],
    cache_key: str,
    min_batch: int,
    max_batch: int,
    auto_batch_cfg: dict[str, Any],
) -> int | None:
    entries = cache_payload.get("entries")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(cache_key)
    if not isinstance(entry, dict):
        return None

    max_age_hours = float(auto_batch_cfg.get("cache_max_age_hours", 168))
    if max_age_hours > 0:
        created_at = entry.get("updated_at_utc")
        if isinstance(created_at, str):
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600.0
                if age_hours > max_age_hours:
                    return None
            except Exception:
                return None

    batch_value = int(entry.get("batch", 0) or 0)
    if batch_value < min_batch or batch_value > max_batch:
        return None
    return batch_value


def _write_cached_batch(
    *,
    cache_path: Path,
    cache_payload: dict[str, Any],
    cache_key: str,
    cache_key_payload: dict[str, Any],
    selected_batch: int,
    best_fit: int,
    attempts: int,
    auto_batch_cfg: dict[str, Any],
) -> None:
    entries = cache_payload.get("entries")
    if not isinstance(entries, dict):
        entries = {}
        cache_payload["entries"] = entries

    entries[cache_key] = {
        "batch": int(selected_batch),
        "best_fit": int(best_fit),
        "attempts": int(attempts),
        "algo_version": AUTO_BATCH_ALGO_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "key_payload": cache_key_payload,
    }

    max_entries = max(10, int(auto_batch_cfg.get("cache_max_entries", 200)))
    if len(entries) > max_entries:
        sortable: list[tuple[str, float]] = []
        for key, item in entries.items():
            timestamp_value = 0.0
            if isinstance(item, dict):
                timestamp_raw = item.get("updated_at_utc")
                if isinstance(timestamp_raw, str):
                    try:
                        timestamp_dt = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                        timestamp_value = timestamp_dt.timestamp()
                    except Exception:
                        timestamp_value = 0.0
            sortable.append((key, timestamp_value))
        sortable.sort(key=lambda item: item[1], reverse=True)
        keep_keys = {key for key, _ in sortable[:max_entries]}
        cache_payload["entries"] = {
            key: value for key, value in entries.items() if key in keep_keys
        }

    _save_cache(cache_path, cache_payload)


def _round_down_to_multiple(value: int, multiple: int, minimum: int) -> int:
    if multiple <= 1:
        return max(minimum, value)
    rounded = (value // multiple) * multiple
    if rounded < minimum:
        return minimum
    return rounded


def _create_synthetic_probe_state(
    *,
    model_name: str,
    num_classes: int,
    image_size: int,
    learning_rate: float,
    device: Any,
    compile_value: Any,
    auto_batch_cfg: dict[str, Any],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    _clear_cuda_cache()
    torch_device = _resolve_torch_device(device)
    if torch_device.type != "cuda":
        raise RuntimeError("Synthetic auto-batch probe requires CUDA.")

    respect_compile = bool(auto_batch_cfg.get("respect_compile", True))
    compile_mode = str(auto_batch_cfg.get("compile_mode", "default"))
    compile_backend = str(auto_batch_cfg.get("compile_backend", "inductor"))
    compile_dynamic = bool(auto_batch_cfg.get("compile_dynamic", False))
    probe_amp = bool(auto_batch_cfg.get("probe_amp", True))
    warmup_steps = max(0, int(auto_batch_cfg.get("synthetic_warmup_steps", 1)))
    measure_steps = max(1, int(auto_batch_cfg.get("synthetic_measure_steps", 1)))
    max_vram_utilization = float(auto_batch_cfg.get("max_vram_utilization", 0.9))
    max_vram_utilization = min(max(max_vram_utilization, 0.5), 0.98)
    max_vram_metric = str(auto_batch_cfg.get("max_vram_metric", "allocated")).strip().lower()
    if max_vram_metric not in {"allocated", "reserved"}:
        max_vram_metric = "allocated"

    raw_model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    ).to(torch_device)
    raw_model.train()
    model = raw_model

    if respect_compile and _is_compile_enabled(compile_value):
        compile_error: BaseException | None = None
        if compile_dynamic:
            try:
                model = torch.compile(
                    raw_model,
                    mode=compile_mode,
                    backend=compile_backend,
                    dynamic=True,
                )
            except TypeError:
                model = raw_model
            except Exception as exc:
                compile_error = exc
                model = raw_model

        if model is raw_model:
            try:
                model = torch.compile(
                    raw_model,
                    mode=compile_mode,
                    backend=compile_backend,
                )
            except Exception as exc:
                compile_error = exc
                if bool(auto_batch_cfg.get("fallback_disable_compile_on_error", True)):
                    model = raw_model
                    print(
                        "[batch-auto] torch.compile probe unavailable "
                        f"({exc.__class__.__name__}). Falling back to eager synthetic probe."
                    )
                else:
                    raise RuntimeError("torch.compile failed for synthetic auto-batch probe") from exc

        if model is raw_model and compile_error is not None:
            auto_batch_cfg["_compile_probe_error"] = (
                f"{compile_error.__class__.__name__}: {compile_error}"
            )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=learning_rate)
    autocast_enabled = probe_amp and amp_enabled and torch_device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=autocast_enabled and amp_dtype == torch.float16,
    )

    total_vram_bytes = int(torch.cuda.get_device_properties(torch_device).total_memory)
    vram_hard_cap_bytes = int(total_vram_bytes * max_vram_utilization)
    return {
        "torch_device": torch_device,
        "model": model,
        "raw_model": raw_model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scaler": scaler,
        "autocast_enabled": autocast_enabled,
        "amp_dtype": amp_dtype,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "warmup_done": False,
        "num_classes": num_classes,
        "image_size": image_size,
        "total_vram_bytes": total_vram_bytes,
        "vram_hard_cap_bytes": vram_hard_cap_bytes,
        "max_vram_utilization": max_vram_utilization,
        "max_vram_metric": max_vram_metric,
    }


def _release_synthetic_probe_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    for key in ("model", "raw_model", "criterion", "optimizer", "scaler"):
        if key in state:
            del state[key]
    state.clear()
    _clear_cuda_cache()


def _probe_batch_fit_synthetic(
    *,
    batch_size: int,
    probe_state: dict[str, Any],
) -> bool:
    torch_device: torch.device = probe_state["torch_device"]
    model: nn.Module = probe_state["model"]
    criterion: nn.Module = probe_state["criterion"]
    optimizer: torch.optim.Optimizer = probe_state["optimizer"]
    scaler: torch.amp.GradScaler = probe_state["scaler"]
    autocast_enabled = bool(probe_state["autocast_enabled"])
    amp_dtype: torch.dtype = probe_state["amp_dtype"]
    num_classes = int(probe_state["num_classes"])
    image_size = int(probe_state["image_size"])
    warmup_steps = int(probe_state["warmup_steps"])
    measure_steps = int(probe_state["measure_steps"])
    vram_hard_cap_bytes = int(probe_state.get("vram_hard_cap_bytes", 0) or 0)
    total_vram_bytes = int(probe_state.get("total_vram_bytes", 0) or 0)
    max_vram_metric = str(probe_state.get("max_vram_metric", "allocated"))
    steps = measure_steps + (0 if bool(probe_state.get("warmup_done", False)) else warmup_steps)

    try:
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(torch_device)

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)

            images = torch.randn(
                int(batch_size),
                3,
                image_size,
                image_size,
                device=torch_device,
                dtype=torch.float32,
            )
            labels = torch.randint(
                low=0,
                high=num_classes,
                size=(int(batch_size),),
                device=torch_device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=autocast_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize(torch_device)
            peak_bytes = int(torch.cuda.max_memory_allocated(torch_device))
            if max_vram_metric == "reserved":
                peak_bytes = int(torch.cuda.max_memory_reserved(torch_device))
            if vram_hard_cap_bytes > 0 and peak_bytes > vram_hard_cap_bytes:
                cap_gb = vram_hard_cap_bytes / (1024**3)
                peak_gb = peak_bytes / (1024**3)
                total_gb = total_vram_bytes / (1024**3) if total_vram_bytes > 0 else 0.0
                print(
                    "[batch-auto] VRAM cap exceeded "
                    f"(metric={max_vram_metric}, peak={peak_gb:.2f} GiB, "
                    f"cap={cap_gb:.2f} GiB, total={total_gb:.2f} GiB)"
                )
                return False

            del images
            del labels
            del logits
            del loss

        probe_state["warmup_done"] = True
        return True
    except RuntimeError as exc:
        if _is_oom_error(exc):
            return False
        raise
    finally:
        _clear_cuda_cache()


def resolve_smart_batch_size(
    *,
    model_name: str,
    num_classes: int,
    image_size: int,
    learning_rate: float,
    device: Any,
    compile_value: Any,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    auto_batch_cfg: dict[str, Any],
) -> int:
    min_batch = max(1, int(auto_batch_cfg.get("min_batch", 2)))
    max_batch = max(min_batch, int(auto_batch_cfg.get("max_batch", 128)))
    start_batch = int(auto_batch_cfg.get("start_batch", 8))
    start_batch = max(min_batch, min(max_batch, start_batch))
    growth_factor = max(2, int(auto_batch_cfg.get("growth_factor", 2)))
    safety_factor = float(auto_batch_cfg.get("safety_factor", 0.9))
    safety_factor = min(max(safety_factor, 0.5), 1.0)
    multiple_of = max(1, int(auto_batch_cfg.get("multiple_of", 2)))
    max_probes = max(4, int(auto_batch_cfg.get("max_probes", 10)))

    torch_device = _resolve_torch_device(device)
    if torch_device.type != "cuda":
        print(f"[batch-auto] CUDA niedostepne. Uzywam start_batch={start_batch}.")
        return start_batch

    use_cache = _cache_enabled_compiled(auto_batch_cfg, compile_value=compile_value)
    cache_path: Path | None = None
    cache_payload: dict[str, Any] | None = None
    cache_key: str | None = None
    cache_key_payload: dict[str, Any] | None = None

    if use_cache:
        cache_path = _resolve_cache_path(auto_batch_cfg)
        cache_payload = _load_cache(cache_path)
        cache_key, cache_key_payload = _batch_cache_key(
            model_name=model_name,
            num_classes=num_classes,
            image_size=image_size,
            device=device,
            compile_value=compile_value,
            auto_batch_cfg=auto_batch_cfg,
            min_batch=min_batch,
            max_batch=max_batch,
            multiple_of=multiple_of,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        cached_batch = _read_cached_batch(
            cache_payload=cache_payload,
            cache_key=cache_key,
            min_batch=min_batch,
            max_batch=max_batch,
            auto_batch_cfg=auto_batch_cfg,
        )
        if cached_batch is not None:
            print(f"[batch-auto] Using cached batch={cached_batch} for compiled profile.")
            return cached_batch

    reuse_synthetic_state = bool(auto_batch_cfg.get("reuse_synthetic_state", True))
    synthetic_probe_state: dict[str, Any] | None = None
    if reuse_synthetic_state:
        synthetic_probe_state = _create_synthetic_probe_state(
            model_name=model_name,
            num_classes=num_classes,
            image_size=image_size,
            learning_rate=learning_rate,
            device=device,
            compile_value=compile_value,
            auto_batch_cfg=auto_batch_cfg,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        max_util = float(synthetic_probe_state.get("max_vram_utilization", 0.0))
        print(
            "[batch-auto] Synthetic probe session initialized "
            f"(reuse=true, max_vram_utilization={max_util:.2f})"
        )

    print(
        "[batch-auto] Smart batch search "
        f"(start={start_batch}, min={min_batch}, max={max_batch}, probes<={max_probes})"
    )

    attempts = 0
    lower_fit = 0
    upper_fail = max_batch + 1

    def probe(candidate: int) -> bool:
        nonlocal attempts, synthetic_probe_state
        attempts += 1
        print(f"[batch-auto] Probe {attempts}/{max_probes}: batch={candidate}")

        state = synthetic_probe_state
        own_state = state is None
        if own_state:
            state = _create_synthetic_probe_state(
                model_name=model_name,
                num_classes=num_classes,
                image_size=image_size,
                learning_rate=learning_rate,
                device=device,
                compile_value=compile_value,
                auto_batch_cfg=auto_batch_cfg,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

        try:
            assert state is not None
            fits = _probe_batch_fit_synthetic(batch_size=candidate, probe_state=state)
            print(f"[batch-auto] batch={candidate} -> {'fits' if fits else 'no-fit'}")
            if not fits and synthetic_probe_state is not None:
                _release_synthetic_probe_state(synthetic_probe_state)
                synthetic_probe_state = _create_synthetic_probe_state(
                    model_name=model_name,
                    num_classes=num_classes,
                    image_size=image_size,
                    learning_rate=learning_rate,
                    device=device,
                    compile_value=compile_value,
                    auto_batch_cfg=auto_batch_cfg,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            return fits
        finally:
            if own_state:
                _release_synthetic_probe_state(state)

    try:
        candidate = start_batch
        if probe(candidate):
            lower_fit = candidate
        else:
            upper_fail = candidate
            while attempts < max_probes and candidate > min_batch:
                candidate = max(min_batch, candidate // growth_factor)
                if probe(candidate):
                    lower_fit = candidate
                    break
                upper_fail = candidate

            if lower_fit == 0:
                print(f"[batch-auto] No fitting probe found above min. Using min_batch={min_batch}")
                return min_batch

        candidate = lower_fit
        while attempts < max_probes and candidate < max_batch and upper_fail == max_batch + 1:
            next_candidate = min(max_batch, candidate * growth_factor)
            if next_candidate == candidate:
                break
            if probe(next_candidate):
                lower_fit = next_candidate
                candidate = next_candidate
            else:
                upper_fail = next_candidate
                break

        while attempts < max_probes and upper_fail - lower_fit > 1:
            mid = (lower_fit + upper_fail) // 2
            if mid <= lower_fit:
                break
            if probe(mid):
                lower_fit = mid
            else:
                upper_fail = mid

        best_fit = max(lower_fit, min_batch)
        safe_batch = int(best_fit * safety_factor)
        safe_batch = _round_down_to_multiple(safe_batch, multiple_of, min_batch)
        safe_batch = min(safe_batch, best_fit)

        print(
            "[batch-auto] Selected "
            f"best_fit={best_fit}, safety_factor={safety_factor}, final_batch={safe_batch}"
        )

        if (
            use_cache
            and cache_path is not None
            and cache_payload is not None
            and cache_key is not None
            and cache_key_payload is not None
        ):
            _write_cached_batch(
                cache_path=cache_path,
                cache_payload=cache_payload,
                cache_key=cache_key,
                cache_key_payload=cache_key_payload,
                selected_batch=safe_batch,
                best_fit=best_fit,
                attempts=attempts,
                auto_batch_cfg=auto_batch_cfg,
            )
            print(f"[batch-auto] Cached compiled batch in {cache_path}")

        return safe_batch
    finally:
        _release_synthetic_probe_state(synthetic_probe_state)
