---
name: walk3d-reconstruction
description: Develop, evaluate, debug, and extend the Walk-to-3D personal video-to-3D reconstruction project using Depth Anything 3 (DA3), including fine-tuned checkpoints, frame selection, multiview fusion, game-ready asset export, scene semantics, and reproducible local CUDA inference. Use for any change in this repository's walk3d package, DA3 integration, model-training/evaluation pipeline, reconstruction quality, or reconstruction-to-game-map workflow.
---

# Walk-to-3D Reconstruction

Build a private, reproducible 3D reconstruction tool from phone walkthroughs. Prioritize reliable reconstruction quality and measured improvements over feature count.

## Read the Project First

Before editing, inspect `README.md`, `docs/findings.md`, `pyproject.toml`, and the affected module in `walk3d/`. Preserve user changes and do not alter files outside the requested scope.

Current architecture:

- `walk3d/frames.py`: video decoding and keyframe choice.
- `walk3d/recon.py`: DA3 loading, inference, depth unprojection, and point selection.
- `walk3d/pipeline.py`: orchestration and artifact metadata.
- `walk3d/ply.py`: binary PLY export.
- `walk3d/app.py`: local Gradio interface.
- `third_party/da3/`: upstream dependency; keep project-specific behavior in `walk3d/` unless intentionally updating the vendor.

## Working Rules

- Treat DA3-BASE as a baseline, not the permanent model.
- Never claim metric accuracy, mesh quality, or general VRAM/runtime without a documented benchmark.
- Keep all model provenance: source checkpoint, base model, dataset version, training code commit, and inference configuration.
- Keep source and UI text UTF-8. Do not introduce machine-specific absolute paths; use configuration or environment variables with safe project-relative defaults.
- Do not overwrite prior reconstructions. Create a unique job directory and record all inputs, outputs, settings, model ID, and timings in `stats.json`.
- Test a narrow change at the unit level, then run a representative CUDA inference when the change affects model execution or geometry.

## Model and Fine-Tuning Workflow

1. Add a model configuration layer before changing inference. It must select `base`, Hugging Face model ID, or local fine-tuned checkpoint without editing source.
2. Cache the loaded model by `(model_id, device, precision)`; never reload it per Gradio request.
3. Keep the base checkpoint as a benchmark control.
4. Before training, define a held-out evaluation split of target scenes. Prevent duplicate locations, video sequences, or near-identical views from crossing train/validation/test splits.
5. Start with parameter-efficient fine-tuning or a frozen visual encoder plus trainable task layers unless evidence justifies full fine-tuning.
6. Log seed, optimizer, learning-rate schedule, resolutions, frame/window policy, loss weights, GPU, checkpoint cadence, and validation results.
7. Promote a checkpoint only if it improves the held-out suite and does not regress the baseline categories.

Evaluate base and tuned models on the identical input windows. Record: depth validity, multiview reprojection consistency, relative camera-pose drift, reconstruction completeness, outlier rate, runtime, peak VRAM, and visual turntables. Use Chamfer/F-score only where calibrated ground-truth geometry exists.

## Reconstruction Workflow

### Short clips

1. Validate video decode, duration, orientation, FPS, and frame count.
2. Select keyframes using sharpness, temporal spacing, visual novelty, and sufficient parallax. Reject near-duplicates, severe motion blur, and exposure failures.
3. Run DA3 with explicit model, device, precision, and processing resolution recorded in output metadata.
4. Unproject only finite, positive, confidence-qualified depth with valid intrinsics/extrinsics.
5. Fuse points using voxel downsampling and statistical/radius outlier filtering. Prefer deterministic sampling for reproducibility.
6. Export the raw/fused point cloud separately. Generate a mesh only as an explicitly labelled optional post-process.

### Long or looping walkthroughs

Do not pass the whole video as one inference window. Split it into overlapping windows, retain shared keyframes, estimate robust similarity alignment between window outputs, and optimize a pose graph. Add loop closures only after verifying them through geometric and appearance consistency. Save per-window transforms and global alignment diagnostics.

## Product Workflow

Keep the product focused on personal/heritage/architecture reconstruction rather than a generic AI demo. Provide:

- upload guidance for slow, steady, overlapping walkthrough footage;
- a job-specific viewer and downloadable PLY/GLB artifacts;
- camera path, point count, runtime, model/checkpoint, and confidence/filter settings;
- manual crop, scale/reference, and cleanup controls when these features are implemented;
- a clear warning that point clouds are not ground-truth-validated surveys.

Run compute-heavy work in a queued background job. The Gradio handler must report progress and surface actionable errors. Do not share one fixed output path across requests.

## Capture and Scene-Type Policy

Choose and document the intended scene class before tuning the model or designing gameplay. Use separate capture profiles for interiors, heritage/outdoor structures, and small objects; do not assume settings transfer between them.

For heritage or outdoor scenes, require slow movement, 60–80% visual overlap, coverage from multiple distances, and a known-size reference such as an ArUco board or measured feature. Flag sky, people, vehicles, water, reflections, and moving foliage as dynamic/low-confidence regions. Record capture device, lens/zoom, orientation, weather/light, and permission/provenance notes with each source video.

Keep a held-out golden set of approximately 20–30 clips across easy, difficult, indoor, outdoor, low-texture, and dynamic scenes. Never train on it. Use it to decide whether a model, filter, or map-export change is an improvement.

## Reconstruction-to-Game-Map Workflow

Treat reconstruction output as source material for a playable environment, not as a shipped game map. Keep this pipeline explicit:

`video -> frames/depth/poses -> raw point cloud -> cleaned geometry -> authored game asset -> engine scene -> navigation and interaction`

1. Preserve immutable source artifacts and reconstruction metadata. Export an interchange asset such as GLB/glTF after cleaning; PLY remains a research/debug format.
2. Remove dynamic regions before fusion. Use semantic segmentation as an assist, but allow review and correction rather than trusting masks blindly.
3. Produce a clean visual mesh and separate collision mesh. Repair holes only where the geometry is supported by source views; label any artist or generative completion.
4. Create UVs, texture atlases, normals, material assignments, LODs, pivots, scale, and collision proxies. Validate coordinate system and units at the export boundary.
5. Generate or author a navigation mesh from collision geometry. Test that every intended route, spawn point, interaction zone, and camera path is reachable.
6. Build a small vertical slice before generating a whole site: one reconstructed area, one traversal loop, one interaction, and one performance target.

Select a game engine deliberately. Keep the reconstruction core engine-agnostic and write a narrow exporter/import contract for the selected engine. Do not add Unity, Unreal, Godot, Blender, or map-tool dependencies until the user chooses the target engine, target platform, visual style, and license constraints.

## Semantic and ML-Assisted Exploration

Use ML to enrich the map only when it produces inspectable data. Good candidates are semantic masks, landmark/object labels, surface/material classes, point-of-interest proposals, and quality warnings. Store each prediction with its model ID, confidence, source frame/geometry reference, and a human-approved status.

Do not let an LLM or vision model silently invent historical facts, inaccessible areas, routes, or geometry. Present uncertain content as suggestions and require manual approval for published labels, narration, quests, or map markers. Keep factual interpretation and reconstructed geometry separately versioned.

## Asset, License, and Performance Gates

Before importing an external open-source project, record its license, model-weight license, redistribution requirements, attribution, dataset restrictions, and compatibility with the chosen engine/distribution channel. Prefer maintained libraries with a narrow role; avoid copying whole applications into this repository.

For every map asset, record source video permission, reconstruction version, post-processing steps, author, and export version. Set platform-specific budgets for triangle count, texture memory, draw calls, collision complexity, and load time only after choosing the engine and target device. Verify the final playable build, not only the mesh viewer.

## Change Checklist

For every implementation change:

1. State the quality, reliability, or usability hypothesis.
2. Identify affected modules and backward-compatibility risks.
3. Add or update tests for pure functions, file contracts, and invalid input.
4. Run formatting/linting and Python compilation.
5. For geometry/model changes, compare against the saved base-model evaluation suite.
6. Update `README.md` and `docs/findings.md` when commands, known limits, measured results, or output contracts change.
7. Report exact artifacts, model/version, test result, and remaining limitation.

## Do Not Do These

- Do not hard-code `DA3-BASE` when a configurable model selector is available.
- Do not hide import or dependency failures with global module stubs in production paths; use supported optional dependencies or clear feature-gated errors.
- Do not label sparse point clouds as watertight meshes or accurate scans.
- Do not train on an unversioned dataset or replace a checkpoint without a baseline comparison.
- Do not commit videos, checkpoints, virtual environments, caches, or generated reconstruction outputs unless explicitly requested.
