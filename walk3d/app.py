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


def reconstruct_video(video_path: str | None, frames: int, process_res: int):
    if not video_path:
        raise gr.Error("Upload a walkthrough video first.")
    summary = run_video(video_path, OUTPUT, target_frames=frames, process_res=process_res)
    return summary["point_cloud"], str(summary)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Walk→3D") as demo:
        gr.Markdown("# Walk→3D\nA local, single-window first pass from video to a colored PLY point cloud.")
        with gr.Row():
            video = gr.Video(sources=["upload"], label="Walkthrough video")
            with gr.Column():
                frame_count = gr.Slider(8, 16, value=12, step=1, label="Keyframes")
                process_res = gr.Slider(196, 504, value=308, step=28, label="DA3 process resolution")
                run = gr.Button("Reconstruct", variant="primary")
        ply_file = gr.File(label="Download point cloud (PLY)")
        details = gr.Textbox(label="Run details", lines=8)
        run.click(reconstruct_video, [video, frame_count, process_res], [ply_file, details])
        gr.Markdown("This prototype uses one DA3 window. Results may contain drift or incomplete geometry.")
    return demo


if __name__ == "__main__":
    Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    build_app().launch(server_name="0.0.0.0", server_port=7860)
