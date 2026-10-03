"""Small Gradio interface for local walkthrough reconstruction."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TEMP", r"D:\octfest\.tmp")
os.environ.setdefault("TMP", r"D:\octfest\.tmp")
os.environ.setdefault("HF_HOME", r"D:\octfest\.hf-cache")

import gradio as gr

from .pipeline import run_video


OUTPUT = Path(__file__).resolve().parents[1] / "out" / "gradio"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOH_PLY = REPO_ROOT / "out" / "soh-max" / "scene.ply"
DEMO_PLY = SOH_PLY if SOH_PLY.exists() else REPO_ROOT / "out" / "demo" / "scene.ply"
EXAMPLE_VIDEO = REPO_ROOT / "third_party" / "da3" / "assets" / "examples" / "robot_unitree.mp4"


def reconstruct_video(video_path: str | None, frames: int, process_res: int):
    if not video_path:
        raise gr.Error("Upload a walkthrough video first.")
    yield gr.skip(), gr.skip(), "⏳ Sampling sharp frames and running Depth Anything 3…", gr.skip()
    summary = run_video(video_path, OUTPUT, target_frames=frames, process_res=process_res)
    point_cloud = summary["point_cloud"]
    status = (
        "### ✓ 3D scene ready\n"
        f"**{summary['point_count']:,} points**　·　{summary['frame_count']} keyframes　·　"
        f"{summary['inference_runtime_seconds']} s inference"
    )
    yield point_cloud, point_cloud, status, summary


def load_example_video():
    if not EXAMPLE_VIDEO.exists():
        raise gr.Error("The bundled sample video is not available in this checkout.")
    return str(EXAMPLE_VIDEO), "Sample walkthrough loaded — press **Build 3D scene** to process it."


def build_app() -> gr.Blocks:
    custom_css = """
    :root { color-scheme: dark; }
    body, .gradio-container { background: #080d18 !important; }
    .gradio-container { max-width: 1480px !important; padding: 30px 36px 52px !important; }
    .gradio-container p, .gradio-container label { color: #d3dced; }
    #hero { padding: 30px 0 26px; border-bottom: 1px solid #1d2a3d; margin-bottom: 22px; }
    #hero h1 { font-size: clamp(2.3rem, 4vw, 3.8rem); letter-spacing: -0.06em; line-height: 1.02; margin: 12px 0; }
    #hero p { color: #a6b5cb; font-size: 1.08rem; max-width: 720px; }
    .eyebrow { color: #66e3d0; font-size: .76rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .panel { background: linear-gradient(145deg, rgba(19,30,49,.96), rgba(12,20,35,.96)); border: 1px solid #263751 !important; border-radius: 20px !important; padding: 20px !important; box-shadow: 0 18px 48px rgba(0,0,0,.16); }
    .panel h3 { color: #f0f5fb; letter-spacing: -.02em; }
    .step-label { color: #66e3d0 !important; font-size: .76rem !important; font-weight: 700 !important; letter-spacing: .14em !important; text-transform: uppercase; }
    #scene-viewer { border: 1px solid #263751 !important; border-radius: 18px !important; overflow: hidden; }
    #scene-viewer:fullscreen { width: 100vw !important; height: 100vh !important; background: #080d18; border: 0 !important; border-radius: 0 !important; }
    #scene-viewer:fullscreen > div { height: 100vh !important; max-height: none !important; }
    #run-button { min-height: 54px; border-radius: 12px; font-weight: 700; box-shadow: 0 8px 26px rgba(20,184,166,.2); }
    #sample-button { min-height: 42px; border-radius: 11px; }
    #fullscreen-button { min-height: 42px; border-radius: 10px; }
    #run-status { color: #a6b5cb; min-height: 28px; border-radius: 12px; }
    .footer-note { color: #8090a8; font-size: .86rem; }
    @media (max-width: 700px) { .gradio-container { padding: 16px 14px 32px !important; } }
    """
    with gr.Blocks(title="Walk→3D | Video to 3D", css=custom_css, theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate")) as demo:
        gr.Markdown(
            "<div class='eyebrow'>Local 3D reconstruction · Powered by Depth Anything 3</div>"
            "<h1>Walk through.<br>Bring back a 3D scene.</h1>"
            "<p>Turn a short video into a colored point cloud you can explore, inspect, and export.</p>",
            elem_id="hero",
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=6, elem_classes="panel"):
                gr.Markdown("### Add your walkthrough", elem_classes="step-label")
                video = gr.Video(sources=["upload"], label="Drop a video here or browse files", height=270)
                sample_button = gr.Button("▶  Try the sample walkthrough", elem_id="sample-button")
            with gr.Column(scale=5, elem_classes="panel"):
                gr.Markdown("### Reconstruction settings", elem_classes="step-label")
                frame_count = gr.Slider(8, 16, value=12, step=1, label="Keyframes", info="Sharp frames sampled across the video")
                process_res = gr.Slider(196, 504, value=308, step=28, label="Processing resolution", info="Higher can preserve more detail and take longer")
                run = gr.Button("Build 3D scene", variant="primary", elem_id="run-button")
                ready_text = "Sample point cloud loaded below. Upload a video or try the sample walkthrough to build your own scene."
                status = gr.Markdown(ready_text, elem_id="run-status")

        gr.Markdown("### Explore your scene")
        with gr.Row():
            gr.Markdown("Drag to orbit · Scroll to zoom · Right-drag to pan")
            fullscreen = gr.Button("⛶  Full screen", elem_id="fullscreen-button", scale=0)
        viewer = gr.Model3D(
            value=str(DEMO_PLY) if DEMO_PLY.exists() else None,
            label="Interactive 3D point cloud",
            display_mode="point_cloud",
            clear_color=(0.025, 0.04, 0.08, 1.0),
            height=620,
            zoom_speed=1.25,
            pan_speed=1.1,
            elem_id="scene-viewer",
        )
        with gr.Row():
            ply_file = gr.File(label="Export point cloud (PLY)", scale=1)
            with gr.Column(scale=2):
                details = gr.JSON(label="Reconstruction details", open=False)
        gr.Markdown(
            "Prototype preview · Results are point clouds and may contain drift or missing surfaces; "
            "they are not accuracy-validated meshes.",
            elem_classes="footer-note",
        )
        run.click(reconstruct_video, [video, frame_count, process_res], [viewer, ply_file, status, details])
        sample_button.click(load_example_video, outputs=[video, status])
        fullscreen.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="""() => {
                const viewer = document.getElementById('scene-viewer');
                if (!viewer) return [];
                if (document.fullscreenElement) document.exitFullscreen();
                else if (viewer.requestFullscreen) viewer.requestFullscreen();
                return [];
            }""",
        )
    return demo


if __name__ == "__main__":
    Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    build_app().launch(server_name="0.0.0.0", server_port=7860)
