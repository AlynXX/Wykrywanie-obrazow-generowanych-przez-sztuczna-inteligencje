from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import build_transforms
from .model import create_model


def load_model_bundle(checkpoint_path: Path, device: torch.device | None = None):
    runtime_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=runtime_device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    model_name = checkpoint["model_name"]

    model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
    ).to(runtime_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return {
        "device": runtime_device,
        "checkpoint": checkpoint,
        "model": model,
        "class_names": class_names,
        "image_size": image_size,
        "model_name": model_name,
    }


def preprocess_image(image: Image.Image, image_size: int, device: torch.device):
    rgb_image = image.convert("RGB")
    transform = build_transforms(image_size=image_size, train=False)
    tensor = transform(rgb_image).unsqueeze(0).to(device)
    return rgb_image, tensor


def analyze_image_quality(image: Image.Image):
    rgb_image = image.convert("RGB")
    image_array = np.array(rgb_image)
    height, width = image_array.shape[:2]
    min_dimension = min(width, height)
    grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    contrast_score = float(grayscale.std())
    reasons = []

    if min_dimension < 384:
        reasons.append("niska rozdzielczosc")
    if blur_score < 45.0:
        reasons.append("silne rozmycie")
    if contrast_score < 28.0:
        reasons.append("bardzo niski kontrast")

    severity = "ok"
    if len(reasons) >= 2 or min_dimension < 256 or blur_score < 20.0:
        severity = "high"
    elif reasons:
        severity = "medium"

    return {
        "warning": bool(reasons),
        "severity": severity,
        "reasons": reasons,
        "width": int(width),
        "height": int(height),
        "min_dimension": int(min_dimension),
        "blur_score": blur_score,
        "contrast_score": contrast_score,
    }


def resolve_target_layer(model: torch.nn.Module, target_layer_name: str | None):
    named_modules = dict(model.named_modules())
    if target_layer_name:
        if target_layer_name not in named_modules:
            available_layers = ", ".join(list(named_modules.keys())[-20:])
            raise ValueError(
                f"Nie znaleziono warstwy '{target_layer_name}'. Przykladowe warstwy: {available_layers}"
            )
        return target_layer_name, named_modules[target_layer_name]

    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name = name
            last_module = module

    if last_module is None or last_name is None:
        raise ValueError("Grad-CAM wymaga modelu z warstwa Conv2d.")

    return last_name, last_module


def compute_probabilities(model: torch.nn.Module, tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, index = torch.max(probabilities, dim=0)
    return probabilities.detach().cpu(), int(index.item()), float(confidence.item())


def compute_grad_cam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: int | None,
):
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        predicted_index = int(probabilities.argmax(dim=0).item())
        selected_index = predicted_index if target_class is None else target_class

        if selected_index < 0 or selected_index >= probabilities.numel():
            raise ValueError(
                f"target_class={selected_index} jest poza zakresem klas 0..{probabilities.numel() - 1}"
            )

        score = logits[:, selected_index].sum()
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("Nie udalo sie pobrac aktywacji lub gradientow dla Grad-CAM.")

        activation_tensor = activations[0]
        gradient_tensor = gradients[0]
        weights = gradient_tensor.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation_tensor).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value

        return probabilities.detach().cpu(), predicted_index, selected_index, cam
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


def render_grad_cam_overlay(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float,
) -> Image.Image:
    alpha = min(max(alpha, 0.0), 1.0)
    image_rgb = np.array(image.convert("RGB"))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlay)


def save_grad_cam_overlay(
    image: Image.Image,
    cam: np.ndarray,
    output_path: Path,
    alpha: float,
):
    overlay_image = render_grad_cam_overlay(image=image, cam=cam, alpha=alpha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_image.save(output_path)


def predict_with_grad_cam(
    bundle: dict,
    image: Image.Image,
    target_layer_name: str | None = None,
    target_class: int | None = None,
    cam_alpha: float = 0.45,
):
    model = bundle["model"]
    device = bundle["device"]
    class_names = bundle["class_names"]
    image_size = bundle["image_size"]

    rgb_image, tensor = preprocess_image(image=image, image_size=image_size, device=device)
    quality = analyze_image_quality(rgb_image)
    resolved_target_layer_name, target_layer = resolve_target_layer(model, target_layer_name)
    probabilities, predicted_index, selected_index, cam = compute_grad_cam(
        model=model,
        tensor=tensor,
        target_layer=target_layer,
        target_class=target_class,
    )
    overlay_image = render_grad_cam_overlay(image=rgb_image, cam=cam, alpha=cam_alpha)

    probability_pairs = [
        {"label": class_name, "probability": float(probabilities[index].item())}
        for index, class_name in enumerate(class_names)
    ]
    probability_pairs.sort(key=lambda item: item["probability"], reverse=True)

    return {
        "predicted_index": predicted_index,
        "selected_index": selected_index,
        "predicted_label": class_names[predicted_index],
        "selected_label": class_names[selected_index],
        "confidence": float(probabilities[selected_index].item()),
        "probabilities": probability_pairs,
        "target_layer": resolved_target_layer_name,
        "overlay_image": overlay_image,
        "cam": cam,
        "input_image": rgb_image,
        "quality": quality,
    }


def predict_image(bundle: dict, image: Image.Image):
    model = bundle["model"]
    device = bundle["device"]
    class_names = bundle["class_names"]
    image_size = bundle["image_size"]

    rgb_image, tensor = preprocess_image(image=image, image_size=image_size, device=device)
    probabilities, predicted_index, confidence = compute_probabilities(model=model, tensor=tensor)
    probability_pairs = [
        {"label": class_name, "probability": float(probabilities[index].item())}
        for index, class_name in enumerate(class_names)
    ]
    probability_pairs.sort(key=lambda item: item["probability"], reverse=True)

    return {
        "predicted_index": predicted_index,
        "predicted_label": class_names[predicted_index],
        "confidence": confidence,
        "probabilities": probability_pairs,
        "quality": analyze_image_quality(rgb_image),
    }
