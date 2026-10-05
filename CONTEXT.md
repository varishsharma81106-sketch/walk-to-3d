# Walk-to-3D: Compact Project Context

## Purpose

Turn phone walkthrough videos into private 3D reconstructions, then evolve selected reconstructions into explorable game environments. The likely focus is heritage/outdoor architecture (for example, Hampi-style sites), but the system must distinguish indoor, outdoor, and object capture profiles.

## Current Project

- Package: `walk3d/`; local Python/Gradio application.
- Model: upstream Depth Anything 3 (DA3-BASE), run locally on CUDA; user also has a fine-tuned DA3 checkpoint that is not yet tracked or integrated in the repository.
- Input: short video walkthrough.
- Current processing: evenly spaced sharp-frame selection -> one DA3 inference window -> depth/confidence/camera-pose unprojection -> colored PLY point cloud.
- Current outputs: `scene.ply`, `camera_path.json`, `stats.json`.
- Measured smoke test: DA3-BASE ran on an RTX 4050 Laptop GPU. This is not an accuracy or general performance claim.

## Current Limitations

- DA3-BASE is hard-coded; no checkpoint/model selector, model cache, or training provenance.
- Single-window reconstruction; no long-video windowing, global alignment, loop closure, or pose graph.
- Frame selection mostly uses temporal spacing and blur/sharpness, not visual novelty or parallax.
- Point cloud is minimally confidence-filtered and randomly capped; no robust fusion, semantic masking, outlier cleanup, mesh, UVs, textures, or GLB export.
- The Gradio app uses a shared output folder; concurrent jobs can overwrite each other.
- No reproducible held-out benchmark, model card, or training manifest exists.
- `da3_compat.py` uses global stubs for optional dependencies; replace with supported feature-gated dependency handling before product expansion.

## Product Direction

This is not a generic AI demo. Aim for a private reconstruction and exploration tool:

`video -> frames/depth/poses -> raw point cloud -> cleaned geometry -> authored game asset -> engine scene -> navigation and interaction`

Treat reconstructed geometry as source material, not a publishable game map. Preserve raw evidence and clearly label any manual or generative completion.

## Prioritized Roadmap

1. Add configurable base/Hugging Face/local fine-tuned checkpoint loading, model caching, and per-job output folders.
2. Establish a capture protocol and a held-out golden evaluation set of roughly 20–30 clips. Never train on this set.
3. Add an evaluation command that compares the identical clips across base and tuned models: depth validity, multiview consistency, pose drift, completeness, outliers, runtime, peak VRAM, and turntable review.
4. Add semantic masks for dynamic/undesirable regions (sky, people, vehicles, water/reflections, moving foliage), then voxel fusion and outlier removal.
5. Add cleaned PLY and GLB/glTF export. Keep raw PLY separately for debugging.
6. Add overlapping DA3 windows, robust alignment, pose-graph optimization, and verified loop closure for larger scenes.
7. Build a game vertical slice: one reconstructed area, one traversal loop, one interaction, and measurable performance target.
8. Add game-ready asset processing: visual mesh, separate collision mesh, UVs/textures/materials, LODs, scale, navmesh, and engine import.

## Capture Rules for Heritage/Outdoor Scenes

- Use slow movement, 60–80% overlap, multiple distances, and a known-size reference (for example, an ArUco board or measured feature).
- Record device, lens/zoom, orientation, lighting/weather, capture permissions, and location provenance.
- Expect failures from sky, foliage, people, vehicles, water, reflections, low texture, and changing light.

## Fine-Tuning Rules

- Keep DA3-BASE as the control model.
- Version dataset, split by location/video sequence to prevent leakage, log training configuration, and preserve checkpoint provenance.
- Begin with parameter-efficient/frozen-encoder approaches unless evaluation proves full fine-tuning is needed.
- Promote a checkpoint only when it improves the held-out set without unacceptable category regressions.
- Create a private model card for every checkpoint: data, base model, strengths, failures, evaluation, hardware/time cost, and examples.

## Game and ML Rules

- Keep the reconstruction core engine-agnostic. Select Godot, Unity, or Unreal only after deciding target platform, visual style, and distribution/license requirements.
- Use ML for inspectable semantic masks, landmark proposals, material/surface labels, and quality warnings. Store model/version/confidence/source evidence and require human approval for published content.
- Do not let ML invent historical facts, inaccessible routes, labels, map markers, or unsupported geometry.
- Review license and redistribution terms for code, weights, data, and assets before integrating any external open-source project.

## Existing Codex Skill

`skills/walk3d-reconstruction/SKILL.md` is the reusable instruction file for this project. It covers reconstruction, fine-tuning, quality gates, capture policy, and reconstruction-to-game-map workflow. Its packaged form is at `skills/dist/walk3d-reconstruction.skill`.

## Immediate Next Decision

Choose the first game engine and target platform:

- Godot: open-source, lightweight, good fit for an independent local-first project.
- Unity: broad ecosystem and asset tooling.
- Unreal: high-fidelity visualization, heavier pipeline and hardware requirements.

Do not add engine-specific dependencies until this decision is made.
