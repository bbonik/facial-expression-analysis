"""Gradio web demo: live webcam and photo analysis of dimensional facial expressions.

Run locally:  scripts/web_demo.sh   (or: uv run --extra demo python app/app.py)
Deploy:       scripts/deploy_space.sh  (Hugging Face Spaces)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gradio as gr
import numpy as np
from core import DEFAULT_MODEL, MODELS, Engine, LiveState, largest, update_live
from render import (
    circumplex_svg,
    draw_overlay,
    faces_grid,
    fit_badge,
    metrics_card,
)

REPO = "https://github.com/bbonik/facial-expression-analysis"
EXAMPLES = next(
    (p for p in (HERE / "examples", HERE.parent / "data" / "images") if p.is_dir()), None
)
FIGURE = next(
    (p for p in (HERE / "assets" / "behavioural_smile.png",
                 HERE.parent / "reports" / "figures" / "behavioural_smile.png") if p.is_file()),
    None,
)  # fmt: skip

ENGINE = Engine()


# ----------------------------------------------------------------------------- handlers
def _wrap_svg(svg: str) -> str:
    return f'<div class="fea-circumplex-wrap">{svg}</div>'


def live_step(frame, model, smoothing, show_landmarks, state: LiveState | None):
    state = state or LiveState()
    if frame is None:
        return None, _wrap_svg(circumplex_svg()), metrics_card(None, model_name=model), state
    frame = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
    faces = ENGINE.analyze(frame, model)
    face = largest(faces)
    state, smoothed = update_live(state, face, model, smoothing)
    overlay = draw_overlay(
        frame,
        [f.box for f in faces],
        [f.landmarks for f in faces],
        [f.point for f in faces],
        show_landmarks=show_landmarks,
    )
    footer = f"{state.fps:.0f} fps · smoothing {smoothing:.2f}" if state.fps else ""
    card = metrics_card(
        smoothed,
        model_name=model,
        fit=ENGINE.fit(face.distance if face else None),
        footer=footer,
    )
    points = [smoothed] if smoothed else []
    return overlay, _wrap_svg(circumplex_svg(points, state.trail)), card, state


def analyze_photo(image, model, show_landmarks):
    if image is None:
        return None, _wrap_svg(circumplex_svg()), faces_grid([])
    image = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
    faces = ENGINE.analyze(image, model, photo=True)
    overlay = draw_overlay(
        image,
        [f.box for f in faces],
        [f.landmarks for f in faces],
        [f.point for f in faces],
        show_landmarks=show_landmarks,
    )
    rows = [(f.point, fit_badge(*ENGINE.fit(f.distance))) for f in faces]
    if len(faces) == 1:  # a single face gets the full card
        f = faces[0]
        card = metrics_card(f.point, model_name=model, fit=ENGINE.fit(f.distance))
        return overlay, _wrap_svg(circumplex_svg([f.point])), card
    return overlay, _wrap_svg(circumplex_svg([f.point for f in faces])), faces_grid(rows)


# ----------------------------------------------------------------------------- content
HERO = f"""
<div class="fea-hero">
  <span class="fea-pill"><span class="fea-live-dot"></span>Real time · 68 landmarks · runs on CPU</span>
  <h1>Facial Expression <span>Analysis</span></h1>
  <p>Arousal, valence and intensity estimated from the geometry of your face, with an
  interpretable linear model trained on 83k faces of 337 people.</p>
  <div class="fea-links">
    <a href="{REPO}" target="_blank">GitHub</a>
    <a href="{REPO}/blob/v2/MODEL_CARD.md" target="_blank">Model card</a>
    <a href="https://arxiv.org/abs/2103.02854" target="_blank">MorphSet · ICIP 2021</a>
    <a href="https://stefan.winkler.site/Publications/icip2020a.pdf" target="_blank">Frontalization · ICIP 2020</a>
  </div>
  <div class="fea-stats">
    <div class="fea-stat"><b>0.80</b><span>valence CCC</span></div>
    <div class="fea-stat"><b>31 KB</b><span>model size</span></div>
    <div class="fea-stat"><b>~70 µs</b><span>per face after landmarks</span></div>
    <div class="fea-stat"><b>24</b><span>circumplex emotions</span></div>
  </div>
</div>
"""

ABOUT = """
<div class="fea-about">
<h3>How it works</h3>
<div class="fea-steps">
  <div class="fea-step"><i>01</i><b>Landmarks</b><span>dlib detects the face and fits 68 points.</span></div>
  <div class="fea-step"><i>02</i><b>Normalise</b><span>Procrustes removes position, scale and in-plane rotation.</span></div>
  <div class="fea-step"><i>03</i><b>Frontalize</b><span>A linear model maps the shape to a frontal view, removing head pose.</span></div>
  <div class="fea-step"><i>04</i><b>Geometry</b><span>1,275 distances between the 51 non-jaw points, scaled by stable points.</span></div>
  <div class="fea-step"><i>05</i><b>Regress</b><span>PLS with 31 components (≈ the number of facial action units).</span></div>
  <div class="fea-step"><i>06</i><b>Describe</b><span>The arousal–valence angle picks one of 24 words of Russell's circumplex.</span></div>
</div>

<h3>Why a linear model?</h3>
<p>A kernel model is more accurate offline (arousal CCC 0.72 vs 0.63), but the linear model
behaves better on new faces, which is what matters live. Measured on 2,294 neutral faces of
people not seen in training:</p>
<table class="fea-table">
  <tr><th></th><th>PLS(31)</th><th>Kernel ridge</th></tr>
  <tr><td>Neutral face: distance from the centre</td><td><b>0.22</b></td><td>0.34</td></tr>
  <tr><td>Smile (mouth corners only): paths &gt; 20° off the valence axis</td><td><b>0.1%</b></td><td>21%</td></tr>
  <tr><td>Prediction spread across head yaw (arousal)</td><td><b>0.075</b></td><td>0.091</td></tr>
</table>
<p>A linear model maps a facial movement to the same change in output for every face. Switch
models in the <b>Live</b> tab and smile: with PLS the point moves along the valence axis.</p>

<h3>Limitations</h3>
<ul>
  <li>Trained on posed, frontal faces with labels generated by morphing, not on spontaneous expressions.</li>
  <li>Arousal follows mouth opening, so talking reads as aroused.</li>
  <li>Errors usually come from poor landmark fits (hands on the face, extreme mouth shapes); the badge flags faces far from the training data.</li>
  <li>The training labels contain no calm or sleepy expressions.</li>
  <li>For research and demos only, not for decisions about individuals. Expressions are not reliable readouts of inner feelings.</li>
</ul>

<h3>Citation</h3>
<p>V. Vonikakis, D. Neo, S. Winkler. <i>MorphSet: Augmenting categorical emotion datasets with
dimensional affect labels using face morphing.</i> ICIP 2021.<br>
V. Vonikakis, S. Winkler. <i>Identity-invariant facial landmark frontalization for facial
expression analysis.</i> ICIP 2020.</p>
</div>
"""

NOTE = (
    '<div class="fea-note">Frames are processed in memory and never stored. '
    f'Open source on <a href="{REPO}" target="_blank">GitHub</a> · MIT licence.</div>'
)

THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

FORCE_DARK = """
() => {
  document.documentElement.classList.add('dark');
  document.body.classList.add('dark');
  document.querySelector('gradio-app')?.classList.add('dark');
}
"""


# ----------------------------------------------------------------------------- layout
def build() -> gr.Blocks:
    with gr.Blocks(title="Facial Expression Analysis", fill_width=True) as demo:
        gr.HTML(HERO, elem_classes="fea-html-plain")
        with gr.Tabs():
            with gr.Tab("Live", id="live"):
                state = gr.State(LiveState())
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=260):
                        cam = gr.Image(
                            sources=["webcam"], streaming=True, type="numpy", label="Camera",
                            webcam_options=gr.WebcamOptions(mirror=True), elem_classes="fea-cam",
                        )  # fmt: skip
                        model = gr.Radio(
                            list(MODELS), value=DEFAULT_MODEL, label="Model",
                            info="Switch while smiling to compare.",
                        )  # fmt: skip
                        smoothing = gr.Slider(
                            0, 0.9, value=0.3, step=0.05, label="Temporal smoothing",
                            info="Higher is calmer but slower to react.",
                        )  # fmt: skip
                        show_lm = gr.Checkbox(True, label="Show landmarks")
                    with gr.Column(scale=5, min_width=360):
                        live_img = gr.Image(
                            label="Analysis", interactive=False, elem_classes="fea-video",
                            buttons=[],
                        )  # fmt: skip
                    with gr.Column(scale=4, min_width=320):
                        live_circ = gr.HTML(
                            _wrap_svg(circumplex_svg()), elem_classes="fea-html-plain"
                        )
                        live_card = gr.HTML(
                            metrics_card(None, model_name=DEFAULT_MODEL),
                            elem_classes="fea-html-plain",
                        )
                cam.stream(
                    live_step,
                    [cam, model, smoothing, show_lm, state],
                    [live_img, live_circ, live_card, state],
                    stream_every=0.08,
                    time_limit=600,
                    concurrency_limit=4,
                    show_progress="hidden",
                )

            with gr.Tab("Photo", id="photo"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320):
                        photo = gr.Image(
                            sources=["upload", "clipboard", "webcam"], type="numpy",
                            label="Photo",
                        )  # fmt: skip
                        photo_model = gr.Radio(list(MODELS), value=DEFAULT_MODEL, label="Model")
                        photo_lm = gr.Checkbox(True, label="Show landmarks")
                        if EXAMPLES is not None:
                            gr.Examples(
                                examples=sorted(
                                    str(p) for p in EXAMPLES.iterdir()
                                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                                ),
                                inputs=photo, label="Examples",
                            )  # fmt: skip
                    with gr.Column(scale=5, min_width=320):
                        photo_out = gr.Image(label="Analysis", interactive=False)
                    with gr.Column(scale=4, min_width=300):
                        photo_circ = gr.HTML(
                            _wrap_svg(circumplex_svg()), elem_classes="fea-html-plain"
                        )
                        photo_faces = gr.HTML(faces_grid([]), elem_classes="fea-html-plain")
                for trigger in (photo.change, photo_model.change, photo_lm.change):
                    trigger(
                        analyze_photo,
                        [photo, photo_model, photo_lm],
                        [photo_out, photo_circ, photo_faces],
                        show_progress="minimal",
                    )

            with gr.Tab("How it works", id="about"):
                gr.HTML(ABOUT, elem_classes="fea-html-plain")
                if FIGURE is not None:
                    gr.Image(
                        str(FIGURE), label="Synthetic smile on 40 unseen faces",
                        interactive=False, buttons=[],
                    )  # fmt: skip
        gr.HTML(NOTE, elem_classes="fea-html-plain")
    return demo


demo = build()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        theme=THEME,
        css_paths=[HERE / "style.css"],
        js=FORCE_DARK,
        inbrowser=os.environ.get("FEA_OPEN_BROWSER") == "1",
    )
