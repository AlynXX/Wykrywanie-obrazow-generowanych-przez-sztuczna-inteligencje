from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
from PIL import Image

from .inference import load_model_bundle, predict_with_grad_cam

DEFAULT_CHECKPOINT = Path("models/best_model.pt")

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
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def format_result_html(result: dict) -> str:
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
        Wynik analizy
      </div>
      <div style="font-size:2rem; margin-top:8px;">
        <span class="score-strong">{result['predicted_label']}</span>
      </div>
      <div style="margin-top:6px; color:#17324d;">
        Pewnosc modelu: <strong>{result['confidence'] * 100:.2f}%</strong>
      </div>
      <div style="margin-top:14px; color:#50657f;">
        Grad-CAM wskazuje, na ktorych obszarach obrazu model opieral decyzje.
      </div>
      <table style="margin-top:16px; width:100%; border-collapse:collapse; color:#17324d;">
        {''.join(bars)}
      </table>
      <div style="margin-top:14px; color:#50657f;">
        Warstwa wyjasniajaca: <strong>{result['target_layer']}</strong>
      </div>
      {warning_html}
    </div>
    """


def build_interface(bundle: dict):
    title_html = f"""
    <div class="hero-card">
      <h1 class="hero-title">Detektor obrazow AI</h1>
      <p class="hero-subtitle">
        Wgraj obraz, aby sprawdzic, czy model traktuje go jako prawdziwe zdjecie czy syntetyczna falszywke.
        Obok wyniku zobaczysz mape Grad-CAM pokazujaca, gdzie sie skupial.
      </p>
      <div class="pill-row">
        <div class="pill">Model: {bundle['model_name']}</div>
        <div class="pill">Rozmiar wejscia: {bundle['image_size']} px</div>
        <div class="pill">Klasy: {", ".join(bundle['class_names'])}</div>
      </div>
    </div>
    """

    def analyze_image(image: Image.Image, cam_alpha: float):
        if image is None:
            raise gr.Error("Najpierw dodaj obraz do analizy.")

        result = predict_with_grad_cam(
            bundle=bundle,
            image=image,
            cam_alpha=cam_alpha,
        )
        return (
            format_result_html(result),
            result["overlay_image"],
            result["input_image"],
        )

    with gr.Blocks(title="Detektor obrazow AI") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(title_html)
            with gr.Row(equal_height=True):
                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["panel-card"]):
                        input_image = gr.Image(
                            type="pil",
                            label="Obraz wejsciowy",
                            height=420,
                            elem_classes=["upload-zone"],
                        )
                        cam_alpha = gr.Slider(
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
                        result_html = gr.HTML(label="Wynik")
                        with gr.Row():
                            overlay_image = gr.Image(
                                type="pil",
                                label="Grad-CAM",
                                height=300,
                            )
                            original_image = gr.Image(
                                type="pil",
                                label="Podglad obrazu",
                                height=300,
                            )

            analyze_button.click(
                fn=analyze_image,
                inputs=[input_image, cam_alpha],
                outputs=[result_html, overlay_image, original_image],
            )

            gr.Markdown(
                "To narzedzie wspiera decyzje analityczne. Wysoka pewnosc modelu nie jest dowodem absolutnym.",
                elem_classes=["footnote"],
            )

    return demo


def main():
    args = parse_args()
    bundle = load_model_bundle(args.checkpoint)
    demo = build_interface(bundle)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
