import numpy as np
import scipy.ndimage as ni
from skimage.measure import marching_cubes


def wnf_to_mesh(wnf_volume, iso_surface_level=0.5, gradient_threshold=0.1, sigma=0.5,
                filter=True, gradient_direction='ascent'):
    volume_size = wnf_volume.shape[-1]
    wnf_ggm = ni.gaussian_gradient_magnitude(
        wnf_volume, sigma=sigma, mode="nearest")
    voxel_spacing = 1 / (volume_size - 1)
    vmin, vmax = wnf_volume.min(), wnf_volume.max()
    if not (vmin <= iso_surface_level <= vmax):
        iso_surface_level = (vmin + vmax) / 2
    try:
        mc_verts, mc_faces, mc_normals, mc_values = marching_cubes(
            wnf_volume,
            level=iso_surface_level,
            spacing=(voxel_spacing,) * 3,
            gradient_direction=gradient_direction,
            method='lewiner')
    except RuntimeError:
        # No surface found — return empty arrays
        print(f'[WARN] marching_cubes: no surface at iso={iso_surface_level:.4f} '
              f'(volume range [{vmin:.4f}, {vmax:.4f}])')
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
    if not filter:
        return mc_verts, mc_faces
    mc_verts_nn_idx = (mc_verts / voxel_spacing).astype(np.uint32)
    mc_verts_ggm = wnf_ggm[
        mc_verts_nn_idx[:, 0], mc_verts_nn_idx[:, 1], mc_verts_nn_idx[:, 2]]
    is_vert_on_surface = mc_verts_ggm > gradient_threshold

    # is_face_valid = np.ones(len(mc_faces), dtype=np.bool)
    is_face_valid = is_vert_on_surface[mc_faces[:, 0]]
    for i in range(1, 3):
        is_face_valid_i = is_vert_on_surface[mc_faces[:, i]]
        # is_face_valid = is_face_valid & is_face_valid_i
        is_face_valid = is_face_valid | is_face_valid_i

    # delete invalid verts
    raw_valid_faces = mc_faces[is_face_valid]
    raw_valid_vert_idx = np.unique(raw_valid_faces.flatten())
    valid_verts = mc_verts[raw_valid_vert_idx]

    valid_vert_idx = np.arange(len(valid_verts))
    vert_raw_idx_valid_idx_map = np.zeros(len(mc_verts), dtype=mc_faces.dtype)
    vert_raw_idx_valid_idx_map[raw_valid_vert_idx] = valid_vert_idx
    valid_faces = vert_raw_idx_valid_idx_map[raw_valid_faces]
    return valid_verts, valid_faces


def delete_invalid_verts(mc_verts, mc_faces, is_vert_on_surface):
    is_face_valid = np.ones(len(mc_faces), dtype=bool)
    for i in range(3):
        is_face_valid_i = is_vert_on_surface[mc_faces[:, i]]
        is_face_valid = is_face_valid & is_face_valid_i

    raw_valid_faces = mc_faces[is_face_valid]
    raw_valid_vert_idx = np.unique(raw_valid_faces.flatten())
    valid_verts = mc_verts[raw_valid_vert_idx]

    valid_vert_idx = np.arange(len(valid_verts))
    vert_raw_idx_valid_idx_map = np.zeros(len(mc_verts), dtype=mc_faces.dtype)
    vert_raw_idx_valid_idx_map[raw_valid_vert_idx] = valid_vert_idx
    valid_faces = vert_raw_idx_valid_idx_map[raw_valid_faces]
    return valid_verts, valid_faces


def collapse_shell_to_surface(nocs_verts, faces, axis=2):
    """Collapse a thick shell mesh (from MC on thickened WNF) to a thin surface.

    For thin cloth, marching cubes on a thickened WNF produces a closed shell
    with upper+lower+side surfaces. This keeps only the upper half and projects
    the remaining vertices to the median plane, removing the systematic offset.

    Args:
        nocs_verts: (N, 3) vertices in NOCS space [0,1]^3
        faces: (F, 3) triangle indices
        axis: axis perpendicular to the thin surface (default=2, z-axis)

    Returns:
        collapsed_verts: (M, 3) with M < N
        collapsed_faces: (G, 3) re-indexed triangles
    """
    if len(nocs_verts) == 0 or len(faces) == 0:
        return nocs_verts, faces

    z_vals = nocs_verts[:, axis]
    z_mid = np.median(z_vals)

    # Keep faces whose centroid is above (or at) z_mid
    face_centroids_z = nocs_verts[faces, axis].mean(axis=1)
    upper_mask = face_centroids_z >= z_mid
    upper_faces = faces[upper_mask]

    if len(upper_faces) == 0:
        return nocs_verts, faces

    # Extract vertices used by upper faces and re-index
    used_ids = np.unique(upper_faces.flatten())
    new_verts = nocs_verts[used_ids].copy()
    new_verts[:, axis] = z_mid  # flatten to median plane

    id_map = np.full(len(nocs_verts), -1, dtype=faces.dtype)
    id_map[used_ids] = np.arange(len(used_ids), dtype=faces.dtype)
    new_faces = id_map[upper_faces]

    print(f'[collapse_shell] {len(nocs_verts)} -> {len(new_verts)} verts, '
          f'{len(faces)} -> {len(new_faces)} faces (z_mid={z_mid:.4f})')

    return new_verts, new_faces
