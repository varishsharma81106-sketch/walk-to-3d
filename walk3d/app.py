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


def reconstruct_video(video_path: str | None, frames: int, process_res: int):
    if not video_path:
        raise gr.Error("Upload a walkthrough video first.")
    summary = run_video(video_path, OUTPUT, target_frames=frames, process_res=process_res)
    point_cloud = summary["point_cloud"]
    return point_cloud, point_cloud, str(summary)


def build_app() -> gr.Blocks:
    custom_css = """
    #scene-viewer:fullscreen { width: 100vw !important; height: 100vh !important; background: #080d18; }
    #scene-viewer:fullscreen > div { height: 100vh !important; max-height: none !important; }
    """
    with gr.Blocks(title="Walk→3D", css=custom_css) as demo:
        gr.Markdown("# Walk→3D\nA local, single-window first pass from video to a colored PLY point cloud.")
        with gr.Row():
            video = gr.Video(sources=["upload"], label="Walkthrough video")
            with gr.Column():
                frame_count = gr.Slider(8, 16, value=12, step=1, label="Keyframes")
                process_res = gr.Slider(196, 504, value=308, step=28, label="DA3 process resolution")
                run = gr.Button("Reconstruct", variant="primary")
        viewer = gr.Model3D(
            value=str(DEMO_PLY) if DEMO_PLY.exists() else None,
            label="Interactive point cloud — drag to orbit, scroll to zoom",
            display_mode="point_cloud",
            clear_color=(0.025, 0.04, 0.08, 1.0),
            height=650,
            zoom_speed=1.25,
            pan_speed=1.1,
            elem_id="scene-viewer",
        )
        with gr.Row():
            fullscreen = gr.Button("⛶ Full screen")
            ply_file = gr.File(label="Download point cloud (PLY)")
        details = gr.Textbox(label="Run details", lines=8)
        run.click(reconstruct_video, [video, frame_count, process_res], [viewer, ply_file, details])
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
        gr.Markdown("This prototype uses one DA3 window. Results may contain drift or incomplete geometry.")
    return demo


if __name__ == "__main__":
    Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    build_app().launch(server_name="0.0.0.0", server_port=7860)
