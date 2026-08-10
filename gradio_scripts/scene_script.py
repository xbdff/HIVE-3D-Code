import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
scripts_dir = os.path.join(project_root, "scripts")
helper_dir = os.path.join(project_root, "hive-3d")
for import_dir in (project_root, scripts_dir, helper_dir):
    if import_dir not in sys.path:
        sys.path.insert(0, import_dir)

os.environ["SPCONV_ALGO"] = "native"

from pathlib import Path
import imageio
from PIL import Image
import torch
import argparse
import tempfile
import zipfile
import open3d as o3d
from trellis.pipelines.test_pipeline import TestImageToSlatPipeline
from trellis.modules.registration.teaser_pp import TeaserRegistrator
from trellis.utils import render_utils
from trellis.modules import sparse as sp
from hive_3d.test_o3d import execute_global_registration, prepare_dataset, execute_local_refinement
from hive_3d.helper import (
    density,
    find_dark_patches_fixed_size,
    find_relevant_voxels_by_mask_path,
    refine_voxel_alignment,
    extract_best_scoring_component,
    extract_largest_component_upsampled,
    save_selected_voxel,
    remove_indices_from_sparsetensor,
    upsample_coords_adaptive,
)
from gradio_scripts.scene_config import (
    SceneConfigError,
    load_scene_config,
    validate_scene_assets,
)


def generate_scene_components(scene: str):
    torch.manual_seed(seed=1)
    # Load an image
    image = Image.open(results_dir / f"{scene}.png")
    reference_cond = pipeline.get_cond([image])

    coords = pipeline.sample_sparse_structure(
        reference_cond, sampler_params=pipeline.sparse_structure_sampler_params
    )

    print(f"Sampled {coords.shape[0]} sparse structure voxels for scene {scene}.")

    gt_slat, ori_attn_dict = pipeline.sample_slat_attention_inpaint(
        reference_cond, coords, pipeline.slat_sampler_params
    )

    ori_attn_map = torch.stack(ori_attn_dict["maps"], dim=0).mean(0)

    Dcoords = pipeline.downsample_coords(coords)

    remove_indices_set = set()
    # Process child nodes.
    for label in scene_tree[scene]:
        #patch_indices = find_dark_patches_fixed_size(results_dir / f"{label}_mask.png")
        #print(len(patch_indices))
        # Match source voxels.
        image = Image.open(results_dir / f"{scene}.png")
        # voxel_indices[label] = pipeline.find_relevant_voxels_by_patches(reference_image = image,attn_map = ori_attn_map,patch_indices=patch_indices,threshold=0.25)
        voxel_score[label], voxel_indices[label] = find_relevant_voxels_by_mask_path(
            mask_path=results_dir / f"{label}_mask.png", attn_map=ori_attn_map, threshold=0.2
        )
        # Filter voxels by connectivity.
        matching_indices[label] = refine_voxel_alignment(
            coords=Dcoords,
            indices=voxel_indices[label],
            k=16,
            fill_threshold=0.6,
            clear_threshold=0.4,
        )
        matching_indices[label] = extract_best_scoring_component(
            coords=Dcoords,
            initial_indices=matching_indices[label],
            voxel_scores=voxel_score[label],
            grid_size=16,
        )
        # Upsample voxels to the original resolution.
        local = pipeline.upsample_coords(
            Dcoords[matching_indices[label]], coords, factor=(2, 2, 2)
        )
        local = extract_largest_component_upsampled(
            orig_coords=coords, matched_indices=local[1], grid_size=64, connectivity=1
        )
        coords_tree[label] = local[0]
        indices_tree[label] = local[1]
        save_selected_voxel(
            coords,
            indices_tree[label],
            intermediate_dir / f"selected_voxel_{label}.glb",
        )
        slat_tree[label] = sp.SparseTensor(
            feats=gt_slat.feats[indices_tree[label]],
            coords=gt_slat.coords[indices_tree[label]],
        )
        GS = pipeline.Slat_to_GS(slat=slat_tree[label], formats=["gaussian"])
        video = render_utils.render_video(GS["gaussian"][0])["color"]
        imageio.mimsave(intermediate_dir / f"ori_{label}.mp4", video, fps=30)

        image = Image.open(results_dir / f"{label}.png")

        # Upsample matched coordinates.
        coords_new = upsample_coords_adaptive(coords_tree[label])
        dense_coords = density(coords_new)
        dense_coords = dense_coords.to(pipeline.device)

        GS_SR = pipeline.run_SR(
            image, seed=1, preprocess_image=False, dense_coords=dense_coords
        )

        video = render_utils.render_video(GS_SR["gaussian"][0])["color"]
        imageio.mimsave(intermediate_dir / f"SR_{label}.mp4", video, fps=30)
        # video = render_utils.render_video(GS_SR['mesh'][0])['normal']
        # imageio.mimsave(intermediate_dir / f"SR_{label}_mesh.mp4", video, fps=30)

        c_SR = (
            GS["gaussian"][0].get_mean_distance_to_center()
            / GS_SR["gaussian"][0].get_mean_distance_to_center()
        )
        print(f"Scale factor for {label}: {c_SR}")
        GS_SR["gaussian"][0].scale_gaussian(scale_factor=c_SR)

        # Save Gaussian point clouds.
        source_path = intermediate_dir / f"source_{label}.ply"
        target_path = intermediate_dir / f"target_{label}.ply"
        GS["gaussian"][0].save_ply_for_open3d(target_path)
        GS_SR["gaussian"][0].save_ply_for_open3d(source_path)
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)

        # RANSAC Registration
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
            source_path,target_path)
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh
                                                    )
        #result_icp= execute_local_refinement(source, target, result_ransac, voxel_size=0.0156)
        #print(result_ransac)
        #print(result_ransac.transformation)
        #source.transform(result_ransac.transformation)
        #result = registrator.solve_registration(source, target)
        GS_SR["gaussian"][0].apply_rigid_transform_to_gaussians(
            result_ransac.transformation
        )
        SR_GS_tree[label] = GS_SR
        remove_indices_set.update(indices_tree[label].tolist())

    remove_indices = list(remove_indices_set)
    if gt_slat.coords.shape[0] == len(remove_indices):
        del SR_GS_tree[scene]
        print(f"Warning: {scene} has no valid data after removing indices, skipping.")
        return
    else:
        gt_slat = remove_indices_from_sparsetensor(gt_slat, remove_indices)
        GS_Origin = pipeline.Slat_to_GS(slat=gt_slat, formats=["gaussian"])
        SR_GS_tree[scene] = GS_Origin
        # Render the outputs
        video = render_utils.render_video(GS_Origin["gaussian"][0])["color"]
        imageio.mimsave(intermediate_dir / f"gt_result_{scene}.mp4", video, fps=30)


parser = argparse.ArgumentParser(description="Run Scene Generation Pipeline")
parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing scene.png, object masks, and config.json")
parser.add_argument("--debug", action="store_true", help="Save intermediate debug artifacts under <input_dir>/debug")
args = parser.parse_args()

results_dir = Path(args.input_dir)
try:
    config = load_scene_config(results_dir / "config.json")
    validate_scene_assets(results_dir, config)
except SceneConfigError as error:
    print(f"Error: {error}", file=sys.stderr)
    raise SystemExit(2) from error

scene_tree = config["scene_tree"]

# Load pipelines only after inexpensive input validation succeeds.
trellis_model = os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS-image-large")
hive3d_model = os.environ.get("HIVE3D_MODEL", "mocun123/HIVE-3D")
pipeline = TestImageToSlatPipeline.from_pretrained(
    trellis_model,
    hive3d_model,
)
pipeline.cuda()

debug_enabled = args.debug
debug_dir = results_dir / "debug"
temp_debug_context = None
if debug_enabled:
    debug_dir.mkdir(exist_ok=True)
    intermediate_dir = debug_dir
else:
    temp_debug_context = tempfile.TemporaryDirectory(prefix="hive3d_scene_")
    intermediate_dir = Path(temp_debug_context.name)

print(f"Processing directory: {results_dir}")
print(f"Debug outputs: {'enabled' if debug_enabled else 'disabled'}")


# Upsampled results.
coords_tree = {}
indices_tree = {}
# Downsampled results.
voxel_score = {}
matching_indices = {}
voxel_indices = {}
# SLAT tensors before and after super-resolution.
slat_tree = {}
SR_GS_tree = {}

image = Image.open(results_dir / "scene.png")
ori_result = pipeline.run(
    image,
    seed=1,
    preprocess_image=False
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# render the result
video = render_utils.render_video(ori_result["gaussian"][0])["color"]
imageio.mimsave(results_dir / "result_trellis.mp4", video, fps=30)
if debug_enabled:
    imageio.mimsave(debug_dir / "ori_scene.mp4", video, fps=30)
# video = render_utils.render_video(ori_result['mesh'][0])['normal']
# imageio.mimsave(debug_dir / "ori_scene_mesh.mp4", video, fps=30)


SR_GS_tree["scene"] = ori_result["gaussian"][0]

generate_scene_components("scene")


# Get the first value; dictionaries preserve insertion order.
first_value = next(iter(SR_GS_tree.values()))
# Merge the remaining values.
rest_values = []
for i, value in enumerate(SR_GS_tree.values()):
    if i > 0:  # Skip the first value.
        first_value["gaussian"][0].merge_gaussians(value["gaussian"][0])


# render the result
video = render_utils.render_video(first_value["gaussian"][0])["color"]
imageio.mimsave(results_dir / "result_hive3d.mp4", video, fps=30)
if debug_enabled and debug_dir.exists():
    debug_zip_path = results_dir / "debug_outputs.zip"
    with zipfile.ZipFile(debug_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in debug_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(results_dir))
    print(f"Debug outputs packaged at: {debug_zip_path}")

if temp_debug_context is not None:
    temp_debug_context.cleanup()
