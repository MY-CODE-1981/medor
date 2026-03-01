"""
Convert fan_2cam_v7 (+ v7_back) Isaac Sim cloth simulation data to medor HDF5 dataset format.

Usage:
    source activate_env.sh
    conda activate softgym
    python scripts/convert_fan_dataset.py

Input (automatically merged):
    - sensor_output_fan_2cam_v7/ (18 traj x 85 frames)
    - sensor_output_fan_2cam_v7_back/ (133 traj x 85 frames)
    - cloth_trajectory_X24Y23_fan/fold/ (NPY, shape=(85,2303,3))
    - cloth_trajectory_X24Y23_fan_v7_back/fold/ (NPY, shape=(85,2303,3))

Output:
    dataset/fan_cloth_v2/{train,val,test}/
        summary_new.h5
        data/00000_3d.h5 ... NNNNN_3d.h5
        nocs/00000_3d.h5
"""

import os
import sys
import argparse
import numpy as np
import cv2
import h5py
from scipy.spatial import cKDTree

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.data_utils import store_h5_dict


# ---------------------------------------------------------------------------
# Camera parameters for fan_2cam_v7_back top camera
# ---------------------------------------------------------------------------
# From run_log.txt: pos=(-0.0025, -0.0523, 0.317)
# focal_length=16.216mm, horizontal_aperture=36.0mm, vertical_aperture=28.685mm
# aperture_offset: h=-0.463081mm, v=-0.08486mm
# Image: 640x480 (VGA), non-square pixels
# fx = 16.216 / 36.0  * 640 = 288.29
# fy = 16.216 / 28.685 * 480 = 271.33
# cx, cy determined empirically (94.5% mask overlap, chamfer=0.006):
#   Isaac Sim depth rendering uses cx = W/2 + h_offset*W/h_ap = 312.0
#   cy = 267.0 (empirically optimal across 6 trajectories)
CAM_POS = np.array([-0.0025, -0.0523, 0.317])
FX = 288.29
FY = 271.33
CX, CY = 312.0, 267.0
DEPTH_SCALE = 1000.0  # uint16 depth values are in mm
IMG_SIZE_ORIG = (640, 480)  # width, height
IMG_SIZE_OUT = 200  # output size for depth, img_nocs, img_pc

# Cloth grid: 47 columns x 49 rows = 2303 vertices
GRID_COLS = 47
GRID_ROWS = 49
NUM_VERTS = GRID_COLS * GRID_ROWS  # 2303

# WNF grid resolution
WNF_RESOLUTION = 128


def make_triangle_faces(rows, cols):
    """Generate triangle faces for a regular grid mesh.

    For a grid of (rows x cols) vertices, generates 2*(rows-1)*(cols-1) triangles.
    Vertex indexing: v[r, c] = r * cols + c
    """
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = r * cols + c
            v01 = r * cols + (c + 1)
            v10 = (r + 1) * cols + c
            v11 = (r + 1) * cols + (c + 1)
            faces.append([v00, v01, v10])
            faces.append([v01, v11, v10])
    return np.array(faces, dtype=np.int32)


def compute_nocs(canon_verts):
    """Normalize canonical vertices to [0,1]^3 via AABB normalization.

    Returns:
        nocs_verts: (N, 3) float32 in [0, 1]^3
        nocs_aabb: (6,) float32 [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    vmin = canon_verts.min(axis=0)
    vmax = canon_verts.max(axis=0)
    extent = vmax - vmin
    max_extent = extent.max()
    # Center and normalize
    center = (vmin + vmax) / 2.0
    nocs_verts = (canon_verts - center) / max_extent + 0.5
    nocs_aabb = np.concatenate([
        center - max_extent / 2.0,
        center + max_extent / 2.0
    ]).astype(np.float32)
    return nocs_verts.astype(np.float32), nocs_aabb


def compute_wnf(nocs_verts, faces, resolution=128, thickness=0.005):
    """Compute Winding Number Field on a regular grid using igl.

    For thin surfaces (like cloth), the mesh is thickened by offsetting vertices
    along normals in both directions to create a closed volume. This gives the
    WNF meaningful "inside" volume for marching cubes extraction.

    Args:
        nocs_verts: (V, 3) float32, vertices in NOCS space
        faces: (F, 3) int32, triangle face indices
        resolution: grid resolution
        thickness: offset distance along normals for thickening (in NOCS units)

    Returns:
        wnf: (resolution, resolution, resolution) float32
    """
    import igl

    V = nocs_verts.astype(np.float64)
    F = faces.astype(np.int32)
    n_verts = V.shape[0]

    # Thicken the mesh: offset vertices along normals in both directions
    normals = igl.per_vertex_normals(V, F)
    verts_top = V + thickness * normals
    verts_bot = V - thickness * normals
    thick_verts = np.concatenate([verts_top, verts_bot], axis=0)
    # Top faces keep original winding; bottom faces reverse winding order
    thick_faces_top = F.copy()
    thick_faces_bot = (F.copy() + n_verts)[:, ::-1]
    thick_faces = np.concatenate([thick_faces_top, thick_faces_bot], axis=0)

    # Create query grid in [0,1]^3
    lin = np.linspace(0, 1, resolution).astype(np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    query_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float64)

    # API varies by igl version
    if hasattr(igl, 'fast_winding_number'):
        wnf_vals = igl.fast_winding_number(thick_verts, thick_faces, query_pts)
    else:
        wnf_vals = igl.fast_winding_number_for_meshes(thick_verts, thick_faces, query_pts)
    wnf = wnf_vals.reshape((resolution, resolution, resolution)).astype(np.float32)
    return wnf


def compute_udf(nocs_verts, faces, resolution=128):
    """Compute Unsigned Distance Field on a regular grid using igl.

    Unlike WNF, UDF is well-defined for open surfaces (no thickening needed).
    The surface is at distance = 0.

    Args:
        nocs_verts: (V, 3) float32, vertices in NOCS space
        faces: (F, 3) int32, triangle face indices
        resolution: grid resolution

    Returns:
        udf: (resolution, resolution, resolution) float32
    """
    import igl

    V = nocs_verts.astype(np.float64)
    F = faces.astype(np.int64)

    # Create query grid in [0,1]^3
    lin = np.linspace(0, 1, resolution).astype(np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    query_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float64)

    # Compute signed distance and take absolute value for UDF
    S, I, C, N = igl.signed_distance(query_pts, V, F)
    udf = np.abs(S).reshape((resolution, resolution, resolution)).astype(np.float32)
    return udf


def backproject_depth(depth_m, mask):
    """Back-project depth image to 3D point cloud in world coordinates.

    The fan_2cam_v7 top camera looks straight down along -Z.
    Convention: world_y = cam_y - y_cam (y-axis flipped).

    Args:
        depth_m: (H, W) float32, depth in meters
        mask: (H, W) bool, valid depth mask

    Returns:
        pc: (N, 3) float32, world coordinates
        uv: (N, 2) int, pixel coordinates (row, col)
    """
    ys, xs = np.where(mask)
    zs = depth_m[ys, xs]

    x_cam = (xs - CX) * zs / FX
    y_cam = (ys - CY) * zs / FY

    world_x = CAM_POS[0] + x_cam
    world_y = CAM_POS[1] - y_cam  # y-axis flipped
    world_z = CAM_POS[2] - zs

    pc = np.stack([world_x, world_y, world_z], axis=-1).astype(np.float32)
    uv = np.stack([ys, xs], axis=-1)
    return pc, uv


def process_frame(depth_path, mask_path, rgb_path, verts_frame, nocs_verts, vert_tree):
    """Process a single frame and return the data dictionary.

    Args:
        depth_path: path to top_depth.png (uint16)
        mask_path: path to top_mask.png (uint8)
        rgb_path: path to top_rgb.png (uint8)
        verts_frame: (2303, 3) float32, vertex positions this frame
        nocs_verts: (2303, 3) float32, canonical NOCS coordinates
        vert_tree: cKDTree built from verts_frame

    Returns:
        dict with keys: depth, pc_sim, pc_nocs, cloth_sim_verts,
                        img_nocs, img_pc, cloth_id, rgb
    """
    # Read images
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)    # uint8
    rgb_raw = cv2.imread(rgb_path, cv2.IMREAD_COLOR)          # BGR uint8
    rgb_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)

    # Convert depth to meters
    depth_m = depth_raw.astype(np.float32) / DEPTH_SCALE
    cloth_mask = mask_raw > 0

    # Full-res point cloud
    pc_full, uv_full = backproject_depth(depth_m, cloth_mask)

    # Assign NOCS to each point via nearest-neighbor to mesh vertices
    _, nn_idx = vert_tree.query(pc_full)
    pc_nocs_full = nocs_verts[nn_idx]

    # Build full-res NOCS image and PC image (640x480)
    h, w = depth_raw.shape[:2]
    img_nocs_full = np.zeros((h, w, 3), dtype=np.float32)
    img_pc_full = np.zeros((h, w, 3), dtype=np.float32)
    img_nocs_full[uv_full[:, 0], uv_full[:, 1]] = pc_nocs_full
    img_pc_full[uv_full[:, 0], uv_full[:, 1]] = pc_full

    # Resize to output size
    sz = IMG_SIZE_OUT
    depth_rs = cv2.resize(depth_m, (sz, sz), interpolation=cv2.INTER_NEAREST)
    img_nocs_rs = cv2.resize(img_nocs_full, (sz, sz), interpolation=cv2.INTER_NEAREST)
    img_pc_rs = cv2.resize(img_pc_full, (sz, sz), interpolation=cv2.INTER_NEAREST)
    rgb_rs = cv2.resize(rgb_raw, (sz, sz), interpolation=cv2.INTER_LINEAR)

    return {
        'depth': depth_rs.astype(np.float32),
        'pc_sim': pc_full.astype(np.float32),
        'pc_nocs': pc_nocs_full.astype(np.float32),
        'cloth_sim_verts': verts_frame.astype(np.float32),
        'img_nocs': img_nocs_rs.astype(np.float32),
        'img_pc': img_pc_rs.astype(np.float32),
        'cloth_id': np.int64(0),
        'rgb': rgb_rs.astype(np.uint8),
    }


def find_matching_trajectories(sensor_dir, vertex_dir):
    """Find trajectory names that exist in both sensor and vertex data.

    Returns list of (traj_name, sensor_dir, vertex_dir) tuples.
    """
    sensor_trajs = set(os.listdir(sensor_dir))
    vertex_files = [f for f in os.listdir(vertex_dir)
                    if f.endswith('.npy') and 'cube' not in f and 'metadata' not in f]
    # vertex files are already named trajectory_angleXX_distXX.npy
    vertex_trajs = set(f.replace('.npy', '') for f in vertex_files)
    common = sorted(sensor_trajs & vertex_trajs)
    return [(name, sensor_dir, vertex_dir) for name in common]


def find_matching_trajectories_multi(sensor_vertex_pairs):
    """Find matching trajectories across multiple sensor/vertex directory pairs.

    Args:
        sensor_vertex_pairs: list of (sensor_dir, vertex_dir) tuples

    Returns:
        list of (traj_name, sensor_dir, vertex_dir) tuples
    """
    all_matches = []
    for sensor_dir, vertex_dir in sensor_vertex_pairs:
        if not os.path.isdir(sensor_dir):
            print(f'  WARNING: sensor dir not found, skipping: {sensor_dir}')
            continue
        if not os.path.isdir(vertex_dir):
            print(f'  WARNING: vertex dir not found, skipping: {vertex_dir}')
            continue
        matches = find_matching_trajectories(sensor_dir, vertex_dir)
        print(f'  {sensor_dir}: {len(matches)} matches')
        all_matches.extend(matches)
    return all_matches


def main():
    parser = argparse.ArgumentParser(description='Convert fan_2cam_v7 to medor HDF5 format')
    parser.add_argument('--data_root',
                        default='/home/initial/.local/share/ov/pkg/isaac-sim-4.5.0/'
                                'cloth_folding_data_collection-main_20251209',
                        help='Root of raw data')
    parser.add_argument('--extra_sensor_dirs', nargs='*', default=[],
                        help='Additional sensor directories (relative to data_root or absolute)')
    parser.add_argument('--extra_vertex_dirs', nargs='*', default=[],
                        help='Additional vertex directories (relative to data_root or absolute)')
    parser.add_argument('--output_dir',
                        default=os.path.join(PROJECT_ROOT, 'dataset', 'fan_cloth_v2'),
                        help='Output dataset directory')
    parser.add_argument('--num_train', type=int, default=120,
                        help='Number of training trajectories')
    parser.add_argument('--num_val', type=int, default=15,
                        help='Number of validation trajectories')
    parser.add_argument('--num_test', type=int, default=16,
                        help='Number of test trajectories')
    parser.add_argument('--skip_wnf', action='store_true',
                        help='Skip WNF computation (requires igl)')
    parser.add_argument('--udf', action='store_true',
                        help='Compute UDF (Unsigned Distance Field) instead of WNF')
    args = parser.parse_args()

    # Build list of (sensor_dir, vertex_dir) pairs
    sensor_vertex_pairs = [
        # v7_back data (133 traj) — use UNFOLD trajectory (matches sensor frame order)
        # Sensor data was rendered from unfold trajectory: frame 0 = folded, frame 84 = flat
        (os.path.join(args.data_root, 'sensor_output_fan_2cam_v7_back'),
         os.path.join(args.data_root, 'cloth_trajectory_X24Y23_fan_v7_back', 'unfold')),
    ]
    # Add any extra dirs from CLI
    for s, v in zip(args.extra_sensor_dirs, args.extra_vertex_dirs):
        s = s if os.path.isabs(s) else os.path.join(args.data_root, s)
        v = v if os.path.isabs(v) else os.path.join(args.data_root, v)
        sensor_vertex_pairs.append((s, v))

    # Find matching trajectories across all pairs
    print('Scanning data directories...')
    traj_entries = find_matching_trajectories_multi(sensor_vertex_pairs)
    traj_names = [e[0] for e in traj_entries]
    print(f'Found {len(traj_entries)} total matching trajectories')

    total_needed = args.num_train + args.num_val + args.num_test
    if len(traj_names) < total_needed:
        print(f'WARNING: Only {len(traj_names)} trajectories available, need {total_needed}')
        print(f'Adjusting splits...')
        args.num_test = min(args.num_test, len(traj_names))
        args.num_val = min(args.num_val, len(traj_names) - args.num_test)
        args.num_train = len(traj_names) - args.num_val - args.num_test

    # Split trajectories (using traj_entries which carry per-traj dirs)
    train_trajs = traj_entries[:args.num_train]
    val_trajs = traj_entries[args.num_train:args.num_train + args.num_val]
    test_trajs = traj_entries[args.num_train + args.num_val:
                              args.num_train + args.num_val + args.num_test]

    print(f'Split: train={len(train_trajs)}, val={len(val_trajs)}, test={len(test_trajs)}')

    # -----------------------------------------------------------------------
    # Step 1: Canonical mesh and NOCS
    # -----------------------------------------------------------------------
    # Use LAST frame of the first trajectory as canonical (flat/unfolded cloth)
    # In the unfold trajectory: frame 0 = folded, frame -1 = flat (unfolded)
    first_name, first_sensor_dir, first_vertex_dir = traj_entries[0]
    verts_path = os.path.join(first_vertex_dir, first_name + '.npy')
    all_verts = np.load(verts_path)
    canon_verts = all_verts[-1]  # last frame = flat cloth (unfolded)
    print(f'Canonical vertices from {first_name} frame {all_verts.shape[0]-1} (last): shape={canon_verts.shape}')

    # Generate triangle faces for 49x47 grid
    faces = make_triangle_faces(GRID_ROWS, GRID_COLS)
    print(f'Generated {len(faces)} triangle faces for {GRID_ROWS}x{GRID_COLS} grid')

    # Compute NOCS
    nocs_verts, nocs_aabb = compute_nocs(canon_verts)
    print(f'NOCS AABB: {nocs_aabb}')

    # -----------------------------------------------------------------------
    # Step 2: WNF or UDF
    # -----------------------------------------------------------------------
    if args.udf:
        print(f'Computing UDF at resolution {WNF_RESOLUTION}^3...')
        wnf = compute_udf(nocs_verts, faces, WNF_RESOLUTION)
        print(f'UDF shape: {wnf.shape}, range: [{wnf.min():.4f}, {wnf.max():.4f}]')
        above_thresh = (wnf < 0.01).sum()
        total = wnf.size
        print(f'  UDF < 0.01 (near-surface): {above_thresh} voxels ({100*above_thresh/total:.2f}%)')
    elif not args.skip_wnf:
        print(f'Computing WNF at resolution {WNF_RESOLUTION}^3...')
        wnf = compute_wnf(nocs_verts, faces, WNF_RESOLUTION)
        print(f'WNF shape: {wnf.shape}, range: [{wnf.min():.4f}, {wnf.max():.4f}]')
    else:
        print('Skipping WNF computation (--skip_wnf). Using placeholder.')
        wnf = np.zeros((WNF_RESOLUTION, WNF_RESOLUTION, WNF_RESOLUTION), dtype=np.float32)

    # -----------------------------------------------------------------------
    # Step 3 & 4: Process each split
    # -----------------------------------------------------------------------
    for split_name, split_trajs in [('train', train_trajs),
                                     ('val', val_trajs),
                                     ('test', test_trajs)]:
        if len(split_trajs) == 0:
            print(f'Skipping empty split: {split_name}')
            continue

        split_dir = os.path.join(args.output_dir, split_name)
        data_dir = os.path.join(split_dir, 'data')
        nocs_dir = os.path.join(split_dir, 'nocs')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(nocs_dir, exist_ok=True)

        # Write NOCS metadata (shared across all frames, cloth_id=0)
        nocs_path = os.path.join(nocs_dir, '00000_3d.h5')
        if not os.path.exists(nocs_path):
            store_h5_dict(nocs_path, {
                'cloth_nocs_verts': nocs_verts,
                'cloth_faces_tri': faces,
                'wnf': wnf,
            })
            print(f'  Wrote {nocs_path}')

        sample_idx = 0
        for traj_name, traj_sensor_base, traj_vertex_base in split_trajs:
            # Load vertex trajectory
            vert_file = traj_name + '.npy'
            vert_path = os.path.join(traj_vertex_base, vert_file)
            traj_verts = np.load(vert_path)  # (85, 2303, 3)
            num_frames = traj_verts.shape[0]

            traj_sensor_dir = os.path.join(traj_sensor_base, traj_name)
            frame_dirs = sorted([d for d in os.listdir(traj_sensor_dir)
                                 if d.startswith('frame_')])

            if len(frame_dirs) != num_frames:
                print(f'  WARNING: {traj_name}: {len(frame_dirs)} frame dirs vs '
                      f'{num_frames} vertex frames. Using min.')
                num_frames = min(len(frame_dirs), num_frames)

            print(f'  [{split_name}] {traj_name}: {num_frames} frames '
                  f'(samples {sample_idx}-{sample_idx + num_frames - 1})')

            for fi in range(num_frames):
                frame_name = f'frame_{fi:04d}'
                frame_dir = os.path.join(traj_sensor_dir, frame_name)

                depth_path = os.path.join(frame_dir, 'top_depth.png')
                mask_path = os.path.join(frame_dir, 'top_mask.png')
                rgb_path = os.path.join(frame_dir, 'top_rgb.png')

                if not os.path.exists(depth_path):
                    print(f'    SKIP: {depth_path} not found')
                    sample_idx += 1
                    continue

                verts_frame = traj_verts[fi]
                vert_tree = cKDTree(verts_frame)

                data = process_frame(
                    depth_path, mask_path, rgb_path,
                    verts_frame, nocs_verts, vert_tree
                )

                out_path = os.path.join(data_dir, f'{sample_idx:05d}_3d.h5')
                store_h5_dict(out_path, data)
                sample_idx += 1

        # Write summary
        summary_path = os.path.join(split_dir, 'summary_new.h5')
        store_h5_dict(summary_path, {
            'nocs_aabb': nocs_aabb,
            'len': np.int64(sample_idx),
            'pos': CAM_POS.astype(np.float32),
            'angle': np.array([0, -np.pi / 2., 0.], dtype=np.float32),
            'width': np.int64(IMG_SIZE_OUT),
            'height': np.int64(IMG_SIZE_OUT),
        })
        print(f'  {split_name}: {sample_idx} samples total')
        print(f'  Wrote {summary_path}')

    print('\nConversion complete!')
    print(f'Output directory: {args.output_dir}')


if __name__ == '__main__':
    main()
