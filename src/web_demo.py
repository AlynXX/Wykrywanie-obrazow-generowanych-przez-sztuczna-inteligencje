from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from .error_analysis import run_error_analysis
from .face_utils import build_face_detector, detect_faces, extract_face_crops, render_face_boxes
from .inference import load_model_bundle, predict_with_grad_cam

DEFAULT_CHECKPOINT = Path("models/best_model.pt")
DEFAULT_FACE_CHECKPOINT = Path("models/faces/best_model.pt")
DEFAULT_CONFIG = Path("config.yaml")

CUSTOM_CSS = """
:root {
  --page-bg: linear-gradient(135deg, #f6f1ea 0%, #e8eef5 100%);
  --panel-bg: rgba(255, 255, 255, 0.86);
  --panel-border: rgba(37, 62, 94, 0.14);
  --ink: #17324d;
  --muted: #50657f;
  --accent: #d66a3d;
  --accent-soft: rgba(214, 106, 61, 0.14);
}

body, .gradio-container {
  background: var(--page-bg) !important;
  color: var(--ink) !important;
  font-family: Georgia, "Times New Roman", serif !important;
}

.gradio-container *,
.gradio-container label,
.gradio-container p,
.gradio-container span,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5,
.gradio-container h6 {
  color: var(--ink) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: var(--ink) !important;
  background: rgba(255, 255, 255, 0.94) !important;
}

.gradio-container button {
  color: var(--ink) !important;
}

.gradio-container button.primary,
.gradio-container button.primary * {
  color: #fffaf5 !important;
}

.upload-zone label {
  color: var(--ink) !important;
}

.upload-zone [data-testid="image"] *,
.upload-zone [data-testid="image"] span,
.upload-zone [data-testid="image"] p,
.upload-zone [data-testid="image"] button,
.upload-zone [data-testid="image"] svg {
  color: #fffaf5 !important;
  fill: #fffaf5 !important;
  stroke: #fffaf5 !important;
}

.app-shell {
  max-width: 1240px;
  margin: 0 auto;
}

.hero-card, .panel-card {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 24px;
  box-shadow: 0 24px 70px rgba(19, 50, 77, 0.09);
  backdrop-filter: blur(10px);
}

.hero-card {
  padding: 28px 30px 22px 30px;
  margin-bottom: 18px;
}

.hero-title {
  font-size: 2.2rem;
  line-height: 1.08;
  margin: 0;
  letter-spacing: -0.03em;
}

.hero-subtitle {
  margin: 10px 0 0 0;
  color: var(--muted);
  font-size: 1rem;
  max-width: 760px;
}

.pill-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 18px;
}

.pill {
  border-radius: 999px;
  padding: 8px 14px;
  background: var(--accent-soft);
  color: var(--ink);
  font-size: 0.95rem;
}

.panel-card {
  padding: 18px;
}

.result-box {
  border-radius: 20px;
  padding: 18px 20px;
  background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(243,248,252,0.92));
  border: 1px solid rgba(37, 62, 94, 0.12);
}

.score-strong {
  color: var(--accent);
  font-weight: 700;
}

.footnote {
  color: var(--muted);
  font-size: 0.92rem;
}

.footnote,
.footnote * {
  color: var(--muted) !important;
}

.warning-box {
  margin-top: 16px;
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(214, 106, 61, 0.12);
  border: 1px solid rgba(214, 106, 61, 0.28);
  color: #7d3b1f;
}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Lokalne demo webowe modelu Real vs AI")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--face-checkpoint",
        type=Path,
        default=DEFAULT_FACE_CHECKPOINT,
        help="Opcjonalny checkpoint modelu twarzowego.",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=None,
        help="Opcjonalny prog decyzyjny dla klasy fake w modelu twarzowym.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def format_result_html(
    result: dict,
    *,
    title: str = "Wynik analizy",
    intro_text: str = "Grad-CAM wskazuje, na ktorych obszarach obrazu model opieral decyzje.",
    extra_html: str = "",
) -> str:
    bars = []
    for item in result["probabilities"]:
        percentage = item["probability"] * 100
        bars.append(
            f"<tr><td style='padding:6px 12px 6px 0;'>{item['label']}</td>"
            f"<td style='padding:6px 0;'>{percentage:.2f}%</td></tr>"
        )

    quality = result.get("quality", {})
    warning_html = ""
    if quality.get("warning"):
        reasons = ", ".join(quality.get("reasons", []))
        warning_html = f"""
        <div class="warning-box">
          <div style="font-weight:700;">Uwaga: obraz ma oznaki niskiej jakosci.</div>
          <div style="margin-top:6px;">
            W takich przypadkach model latwiej myli prawdziwe zdjecia z obrazami AI.
          </div>
          <div style="margin-top:8px;">
            Powody: <strong>{reasons}</strong>
          </div>
          <div style="margin-top:8px;">
            Rozdzielczosc: <strong>{quality['width']} x {quality['height']}</strong>,
            ostrosc: <strong>{quality['blur_score']:.1f}</strong>,
            kontrast: <strong>{quality['contrast_score']:.1f}</strong>
          </div>
        </div>
        """
    return f"""
    <div class="result-box">
      <div style="font-size:0.92rem; color:#50657f; text-transform:uppercase; letter-spacing:0.08em;">
        {title}
      </div>
      <div style="font-size:2rem; margin-top:8px;">
        <span class="score-strong">{result['predicted_label']}</span>
      </div>
      <div style="margin-top:6px; color:#17324d;">
        Pewnosc modelu: <strong>{result['confidence'] * 100:.2f}%</strong>
      </div>
      <div style="margin-top:14px; color:#50657f;">
        {intro_text}
      </div>
      <table style="margin-top:16px; width:100%; border-collapse:collapse; color:#17324d;">
        {''.join(bars)}
      </table>
      <div style="margin-top:14px; color:#50657f;">
        Warstwa wyjasniajaca: <strong>{result['target_layer']}</strong>
      </div>
      {extra_html}
      {warning_html}
    </div>
    """


def format_pending_html(message: str, *, title: str) -> str:
    return f"""
    <div class="result-box">
      <div style="font-size:0.92rem; color:#50657f; text-transform:uppercase; letter-spacing:0.08em;">
        {title}
      </div>
      <div style="margin-top:10px; color:#17324d;">
        {message}
      </div>
    </div>
    """


def detect_primary_face(image: Image.Image, detector) -> tuple[Image.Image | None, Image.Image]:
    image_rgb = image.convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    detections = detect_faces(
        image_bgr,
        detector=detector,
        min_face_size=64,
        scale_factor=1.1,
        min_neighbors=5,
    )
    face_records = extract_face_crops(
        image_bgr,
        detections,
        margin_ratio=0.22,
        square_crop=True,
        selection="largest",
        max_faces=1,
    )
    if not face_records:
        return None, image_rgb

    annotated_bgr = render_face_boxes(image_bgr, face_records)
    annotated_image = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    face_image = Image.fromarray(face_records[0]["image_rgb"])
    return face_image, annotated_image


def format_analysis_html(summary: dict) -> str:
    metrics = summary.get("metrics", {})
    metric_rows = []
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision macro"),
        ("recall_macro", "Recall macro"),
        ("f1_macro", "F1 macro"),
        ("roc_auc", "ROC-AUC"),
    ]:
        if key in metrics:
            metric_rows.append(
                f"<tr><td style='padding:6px 12px 6px 0;'>{label}</td>"
                f"<td style='padding:6px 0;'>{metrics[key]:.4f}</td></tr>"
            )

    group_rows = []
    for group_name, count in summary.get("groups", {}).items():
        group_rows.append(
            f"<tr><td style='padding:6px 12px 6px 0;'>{group_name}</td>"
            f"<td style='padding:6px 0;'>{count}</td></tr>"
        )

    matrix = summary.get("confusion_matrix", {})
    matrix_html = ""
    if matrix:
        class_names = summary.get("class_names", [])
        header_cells = "".join(
            f"<th style='padding:6px 10px; text-align:left;'>{class_name}</th>"
            for class_name in class_names
        )
        body_rows = []
        for actual_name, predicted_counts in matrix.items():
            value_cells = "".join(
                f"<td style='padding:6px 10px;'>{predicted_counts.get(class_name, 0)}</td>"
                for class_name in class_names
            )
            body_rows.append(
                f"<tr><td style='padding:6px 10px; font-weight:700;'>{actual_name}</td>{value_cells}</tr>"
            )
        matrix_html = f"""
        <div style="margin-top:18px;">
          <div style="font-weight:700; margin-bottom:8px;">Macierz pomylek</div>
          <table style="width:100%; border-collapse:collapse;">
            <thead>
              <tr>
                <th style='padding:6px 10px; text-align:left;'>Rzeczywista \\ Przewidziana</th>
                {header_cells}
              </tr>
            </thead>
            <tbody>
              {''.join(body_rows)}
            </tbody>
          </table>
        </div>
        """

    return f"""
    <div class="result-box">
      <div style="font-size:0.92rem; color:#50657f; text-transform:uppercase; letter-spacing:0.08em;">
        Raport analizy bledow
      </div>
      <div style="margin-top:10px; color:#17324d;">
        Split: <strong>{summary['split']}</strong> |
        probki: <strong>{summary['num_samples']}</strong> |
        bledy: <strong>{summary['num_errors']}</strong> |
        filtr klasy: <strong>{summary.get('filter_class', 'all')}</strong> |
        sortowanie: <strong>{summary.get('sort_mode', 'default')}</strong>
      </div>
      <div style="margin-top:14px; display:grid; grid-template-columns: 1fr 1fr; gap:18px;">
        <div>
          <div style="font-weight:700; margin-bottom:8px;">Metryki</div>
          <table style="width:100%; border-collapse:collapse;">{''.join(metric_rows)}</table>
        </div>
        <div>
          <div style="font-weight:700; margin-bottom:8px;">Liczba przypadkow</div>
          <table style="width:100%; border-collapse:collapse;">{''.join(group_rows)}</table>
        </div>
      </div>
      {matrix_html}
    </div>
    """


def build_gallery_items(summary: dict, group_name: str):
    manifest_entries = summary.get("examples_manifest", {}).get(group_name, [])
    gallery_items = []
    for entry in manifest_entries:
        image_path = entry.get("grad_cam_image") or entry.get("copied_image")
        if not image_path:
            continue
        caption = (
            f"GT: {entry['actual_label']} | Pred: {entry['predicted_label']} | "
            f"Conf: {entry['confidence'] * 100:.2f}% | "
            f"Difficulty: {entry.get('difficulty_score', 0.0):.3f}"
        )
        gallery_items.append((image_path, caption))
    return gallery_items


def build_gallery_update(summary: dict, group_name: str):
    items = build_gallery_items(summary, group_name)
    return gr.update(value=items, visible=bool(items))


def build_interface(bundle: dict, face_bundle: dict | None = None):
    face_model_status = (
        f"aktywny: {face_bundle['model_name']}" if face_bundle is not None else "niedostepny"
    )
    face_threshold = face_bundle.get("decision_threshold") if face_bundle is not None else None
    face_threshold_label = (
        f"{face_threshold:.3f}" if isinstance(face_threshold, (float, int)) else "argmax"
    )
    face_detector = build_face_detector() if face_bundle is not None else None
    title_html = f"""
    <div class="hero-card">
      <h1 class="hero-title">Detektor obrazow AI</h1>
      <p class="hero-subtitle">
        Wgraj obraz, aby sprawdzic, czy model traktuje go jako prawdziwe zdjecie czy syntetyczna falszywke.
        Demo pokazuje teraz dwa etapy projektu: model globalny dla calego obrazu i model twarzowy dla wykrytej twarzy.
      </p>
      <div class="pill-row">
        <div class="pill">Model globalny: {bundle['model_name']}</div>
        <div class="pill">Model twarzowy: {face_model_status}</div>
        <div class="pill">Prog twarzowy: {face_threshold_label}</div>
        <div class="pill">Rozmiar wejscia: {bundle['image_size']} px</div>
        <div class="pill">Klasy: {", ".join(bundle['class_names'])}</div>
      </div>
    </div>
    """

    def image_update(value, visible: bool):
        return gr.update(value=value, visible=visible)

    def analyze_image(image: Image.Image, cam_alpha: float):
        if image is None:
            raise gr.Error("Najpierw dodaj obraz do analizy.")

        base_image = image.convert("RGB")
        global_result_html = format_pending_html(
            "Analiza globalna jest gotowa do uzycia dla pelnych kadrow bez dominujacej twarzy.",
            title="Wynik modelu globalnego",
        )
        original_preview = base_image
        face_result_html = format_pending_html(
            "Model twarzowy nie jest zaladowany. Uruchom demo z checkpointem w models/faces/best_model.pt.",
            title="Model twarzowy",
        )
        global_overlay_update = image_update(None, False)
        original_preview_update = image_update(base_image, True)
        face_overlay_update = image_update(None, False)
        face_crop_update = image_update(None, False)

        if face_bundle is not None:
            face_image, annotated_preview = detect_primary_face(base_image, face_detector)
            original_preview_update = image_update(annotated_preview, True)
            if face_image is None:
                global_result = predict_with_grad_cam(
                    bundle=bundle,
                    image=base_image,
                    cam_alpha=cam_alpha,
                )
                global_result_html = format_result_html(
                    global_result,
                    title="Wynik modelu globalnego",
                    intro_text="Grad-CAM wskazuje, na ktorych obszarach calego obrazu model opieral decyzje.",
                )
                global_overlay_update = image_update(global_result["overlay_image"], True)
                original_preview_update = image_update(global_result["input_image"], True)
                face_result_html = format_pending_html(
                    "Nie wykryto twarzy na tym obrazie, wiec analiza twarzowa nie zostala uruchomiona."
                    " Pokazano tylko wynik modelu globalnego.",
                    title="Model twarzowy",
                )
            else:
                global_result_html = format_pending_html(
                    "Wykryto twarz, wiec wynik modelu globalnego zostal ukryty, aby nie mieszac go z bardziej trafna analiza twarzowa.",
                    title="Wynik modelu globalnego",
                )
                face_crop_update = image_update(face_image, True)
                face_result = predict_with_grad_cam(
                    bundle=face_bundle,
                    image=face_image,
                    cam_alpha=cam_alpha,
                )
                face_result_html = format_result_html(
                    face_result,
                    title="Wynik modelu twarzowego",
                    intro_text="Ten wynik pochodzi z osobnego modelu wytrenowanego na cropach twarzy.",
                    extra_html=(
                        "<div style='margin-top:14px; color:#50657f;'>"
                        "Analiza dotyczy najwiekszej wykrytej twarzy na obrazie."
                        "</div>"
                    ),
                )
                face_overlay_update = image_update(face_result["overlay_image"], True)
                face_crop_update = image_update(face_result["input_image"], True)
        else:
            global_result = predict_with_grad_cam(
                bundle=bundle,
                image=base_image,
                cam_alpha=cam_alpha,
            )
            global_result_html = format_result_html(
                global_result,
                title="Wynik modelu globalnego",
                intro_text="Grad-CAM wskazuje, na ktorych obszarach calego obrazu model opieral decyzje.",
            )
            global_overlay_update = image_update(global_result["overlay_image"], True)
            original_preview_update = image_update(global_result["input_image"], True)

        return (
            global_result_html,
            face_result_html,
            global_overlay_update,
            original_preview_update,
            face_overlay_update,
            face_crop_update,
        )

    def run_error_analysis_from_ui(
        split: str,
        filter_class: str,
        sort_mode: str,
        examples_per_group: int,
        save_grad_cam: bool,
        config_path_value: str,
        output_dir_value: str,
        target_layer_name: str,
        cam_alpha: float,
    ):
        config_path = Path(config_path_value.strip() or str(DEFAULT_CONFIG))
        output_dir = Path(output_dir_value.strip() or "reports/error_analysis")
        target_layer = target_layer_name.strip() or None

        try:
            result = run_error_analysis(
                checkpoint_path=Path(bundle.get("checkpoint_path", DEFAULT_CHECKPOINT)),
                config_path=config_path,
                data_dir=None,
                split=split,
                output_dir=output_dir,
                examples_per_group=int(examples_per_group),
                save_grad_cam=save_grad_cam,
                target_layer_name=target_layer,
                cam_alpha=cam_alpha,
                filter_class=filter_class,
                sort_mode=sort_mode,
            )
        except Exception as error:
            raise gr.Error(str(error)) from error

        summary = result["summary"]
        report_text = (
            f"Raport zapisany w: {result['output_root']}\n"
            f"CSV: {result['predictions_csv_path']}\n"
            f"Summary: {result['summary_path']}"
        )
        return (
            format_analysis_html(summary),
            summary,
            str(result["predictions_csv_path"]),
            str(result["summary_path"]),
            report_text,
            build_gallery_update(summary, "false_positive"),
            build_gallery_update(summary, "false_negative"),
            build_gallery_update(summary, "true_positive"),
            build_gallery_update(summary, "true_negative"),
            build_gallery_update(summary, "errors"),
            build_gallery_update(summary, "correct"),
        )

    with gr.Blocks(title="Detektor obrazow AI") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(title_html)
            with gr.Tabs():
                with gr.Tab("Pojedynczy obraz"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=5):
                            with gr.Group(elem_classes=["panel-card"]):
                                input_image = gr.Image(
                                    type="pil",
                                    label="Obraz wejsciowy",
                                    height=420,
                                    elem_classes=["upload-zone"],
                                )
                                cam_alpha_single = gr.Slider(
                                    minimum=0.15,
                                    maximum=0.75,
                                    value=0.45,
                                    step=0.05,
                                    label="Moc nakladki Grad-CAM",
                                )
                                analyze_button = gr.Button("Analizuj obraz", variant="primary")
                                gr.Markdown(
                                    "Model zwroci etykiete, pewnosc i mape ciepla wskazujaca obszary istotne dla decyzji.",
                                    elem_classes=["footnote"],
                                )

                        with gr.Column(scale=4):
                            with gr.Group(elem_classes=["panel-card"]):
                                result_html = gr.HTML(label="Wynik modelu globalnego")
                                with gr.Row():
                                    overlay_image = gr.Image(
                                        type="pil",
                                        label="Grad-CAM modelu globalnego",
                                        height=300,
                                    )
                                    original_image = gr.Image(
                                        type="pil",
                                        label="Podglad obrazu / wykryta twarz",
                                        height=300,
                                    )
                            with gr.Group(elem_classes=["panel-card"]):
                                face_result_html = gr.HTML(label="Wynik modelu twarzowego")
                                with gr.Row():
                                    face_overlay_image = gr.Image(
                                        type="pil",
                                        label="Grad-CAM modelu twarzowego",
                                        height=280,
                                    )
                                    face_crop_image = gr.Image(
                                        type="pil",
                                        label="Crop twarzy",
                                        height=280,
                                    )

                    analyze_button.click(
                        fn=analyze_image,
                        inputs=[input_image, cam_alpha_single],
                        outputs=[
                            result_html,
                            face_result_html,
                            overlay_image,
                            original_image,
                            face_overlay_image,
                            face_crop_image,
                        ],
                    )

                with gr.Tab("Analiza bledow"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=4):
                            with gr.Group(elem_classes=["panel-card"]):
                                split_input = gr.Dropdown(
                                    choices=["test", "val", "train"],
                                    value="test",
                                    label="Split do analizy",
                                )
                                class_filter_input = gr.Dropdown(
                                    choices=["all", *bundle["class_names"]],
                                    value="all",
                                    label="Filtr klasy",
                                )
                                sort_mode_input = gr.Dropdown(
                                    choices=[
                                        "default",
                                        "hardest",
                                        "most_confident",
                                        "least_confident",
                                    ],
                                    value="hardest",
                                    label="Sortowanie przypadkow",
                                )
                                examples_input = gr.Slider(
                                    minimum=1,
                                    maximum=30,
                                    value=12,
                                    step=1,
                                    label="Przykladow na grupe",
                                )
                                save_grad_cam_input = gr.Checkbox(
                                    value=True,
                                    label="Zapisz Grad-CAM dla eksportowanych przykladow",
                                )
                                config_path_input = gr.Textbox(
                                    value=str(DEFAULT_CONFIG),
                                    label="Sciezka config.yaml",
                                )
                                output_dir_input = gr.Textbox(
                                    value="reports/error_analysis",
                                    label="Folder raportu",
                                )
                                target_layer_input = gr.Textbox(
                                    value="",
                                    label="Warstwa Grad-CAM (opcjonalnie)",
                                )
                                cam_alpha_batch = gr.Slider(
                                    minimum=0.15,
                                    maximum=0.75,
                                    value=0.45,
                                    step=0.05,
                                    label="Moc nakladki Grad-CAM",
                                )
                                analyze_errors_button = gr.Button(
                                    "Uruchom analize bledow",
                                    variant="primary",
                                )
                                gr.Markdown(
                                    "Filtr klasy pokazuje przypadki, gdzie dana klasa jest rzeczywista albo przewidziana. Tryb `hardest` promuje najbardziej mylace przypadki.",
                                    elem_classes=["footnote"],
                                )

                        with gr.Column(scale=5):
                            with gr.Group(elem_classes=["panel-card"]):
                                analysis_html = gr.HTML(label="Raport")
                                analysis_json = gr.JSON(label="Summary JSON")
                                predictions_file = gr.File(label="predictions.csv")
                                summary_file = gr.File(label="summary.json")
                                output_info = gr.Textbox(label="Zapisane artefakty")

                    with gr.Group(elem_classes=["panel-card"]):
                        false_positive_gallery = gr.Gallery(
                            label="False positive",
                            visible=False,
                            columns=3,
                            height="auto",
                        )
                        false_negative_gallery = gr.Gallery(
                            label="False negative",
                            visible=False,
                            columns=3,
                            height="auto",
                        )
                        true_positive_gallery = gr.Gallery(
                            label="True positive",
                            visible=False,
                            columns=3,
                            height="auto",
                        )
                        true_negative_gallery = gr.Gallery(
                            label="True negative",
                            visible=False,
                            columns=3,
                            height="auto",
                        )
                        errors_gallery = gr.Gallery(
                            label="Errors",
                            visible=False,
                            columns=3,
                            height="auto",
                        )
                        correct_gallery = gr.Gallery(
                            label="Correct",
                            visible=False,
                            columns=3,
                            height="auto",
                        )

                    analyze_errors_button.click(
                        fn=run_error_analysis_from_ui,
                        inputs=[
                            split_input,
                            class_filter_input,
                            sort_mode_input,
                            examples_input,
                            save_grad_cam_input,
                            config_path_input,
                            output_dir_input,
                            target_layer_input,
                            cam_alpha_batch,
                        ],
                        outputs=[
                            analysis_html,
                            analysis_json,
                            predictions_file,
                            summary_file,
                            output_info,
                            false_positive_gallery,
                            false_negative_gallery,
                            true_positive_gallery,
                            true_negative_gallery,
                            errors_gallery,
                            correct_gallery,
                        ],
                    )

            gr.Markdown(
                "To narzedzie wspiera decyzje analityczne. Wysoka pewnosc modelu nie jest dowodem absolutnym.",
                elem_classes=["footnote"],
            )

    return demo


def main():
    args = parse_args()
    bundle = load_model_bundle(args.checkpoint)
    bundle["checkpoint_path"] = str(args.checkpoint)
    face_bundle = None
    if args.face_checkpoint is not None and args.face_checkpoint.exists():
        face_bundle = load_model_bundle(
            args.face_checkpoint,
            decision_threshold=args.face_threshold,
        )
        face_bundle["checkpoint_path"] = str(args.face_checkpoint)
    demo = build_interface(bundle, face_bundle=face_bundle)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
