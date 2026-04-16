"""
This script provides a naive baseline for FIFA Skeletal Tracking Challenge.

Author: Tianjian Jiang
Date: Nov 10, 2025
"""

from pathlib import Path

import cv2
import numpy as np
import scipy.optimize
import torch
import torch.optim as optim
from tqdm import tqdm

from lib.camera_tracker import CameraTracker, CameraTrackerOptions
from lib.postprocess import smoothen

OPENPOSE_TO_OURS = [0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]

# Connectivity array for the 15-point skeleton subset
# Each pair [i, j] defines a connection between points i and j in the subset
SKELETON_CONNECTIVITY = [
    [0, 1],  # Nose to RShoulder
    [0, 2],  # Nose to LShoulder
    [1, 3],  # RShoulder to RElbow
    [3, 5],  # RElbow to RWrist
    [2, 4],  # LShoulder to LElbow
    [4, 6],  # LElbow to LWrist
    [1, 7],  # RShoulder to RHip
    [2, 8],  # LShoulder to LHip
    [7, 8],  # RHip to LHip
    [7, 9],  # RHip to RKnee
    [9, 11],  # RKnee to RAnkle
    [11, 13],  # RAnkle to RBigToe
    [8, 10],  # LHip to LKnee
    [10, 12],  # LKnee to LAnkle
    [12, 14],  # LAnkle to LBigToe
]


def compute_reprojection_error(predictions, skels_2d, cameras, Rt, boxes, sequences_file=None):
    """Compute reprojection error to diagnose if camera/pose is the bottleneck.

    Args:
        predictions: (NUM_PERSONS, NUM_FRAMES, 15, 3) predicted 3D skeletons
        skels_2d: (NUM_FRAMES, NUM_PERSONS, 15, 2) 2D skeleton detections
        cameras: dict with K, k for all frames
        Rt: list of (R, t) tuples for all frames
        boxes: (NUM_FRAMES, NUM_PERSONS, 4) bounding boxes (for validity check)
        sequences_file: optional file path to report by sequence

    Returns:
        dict with reprojection statistics
    """
    NUM_PERSONS, NUM_FRAMES = predictions.shape[:2]

    # Get camera parameters
    R = np.array([k[0] for k in Rt])  # (NUM_FRAMES, 3, 3)
    t = np.array([k[1] for k in Rt])  # (NUM_FRAMES, 3)

    # Valid mask: frames where all persons have valid boxes
    valid = ~np.isnan(boxes).any(axis=-1).transpose(1, 0)  # (NUM_PERSONS, NUM_FRAMES)

    reprojection_errors = []
    per_player_errors = [[] for _ in range(NUM_PERSONS)]
    per_player_centers = [[] for _ in range(NUM_PERSONS)]

    for frame_idx in range(NUM_FRAMES):
        for person in range(NUM_PERSONS):
            if not valid[person, frame_idx]:
                continue

            skel_3d = predictions[person, frame_idx]  # (15, 3)
            skel_2d_gt = skels_2d[frame_idx, person]  # (15, 2)

            # Skip if any NaN in ground truth
            if np.isnan(skel_2d_gt).any() or np.isnan(skel_3d).any():
                continue

            # Project 3D skeleton to 2D
            pts_2d, _ = cv2.projectPoints(
                skel_3d,
                cv2.Rodrigues(R[frame_idx])[0],
                t[frame_idx],
                cameras["K"][frame_idx],
                cameras["k"][frame_idx],
            )
            pts_2d = pts_2d.squeeze(axis=1)  # (15, 2)

            # Compute L2 distance in pixels
            errors = np.linalg.norm(pts_2d - skel_2d_gt, axis=1)  # (15,)
            reprojection_errors.extend(errors)
            per_player_errors[person].extend(errors.tolist())

            box = boxes[frame_idx, person]
            box_center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
            per_player_centers[person].append(box_center)

    reprojection_errors = np.array(reprojection_errors)
    per_player_stats = []
    for person in range(NUM_PERSONS):
        if not per_player_errors[person]:
            continue
        mean_center = np.mean(per_player_centers[person], axis=0)
        per_player_stats.append(
            {
                "player_id": person,
                "mean_reprojection_error": float(np.mean(per_player_errors[person])),
                "median_reprojection_error": float(np.median(per_player_errors[person])),
                "mean_x": float(mean_center[0]),
                "mean_y": float(mean_center[1]),
                "num_observations": len(per_player_errors[person]),
            }
        )

    per_player_stats.sort(key=lambda stats: stats["mean_reprojection_error"])

    return {
        "mean_reprojection_error": float(np.mean(reprojection_errors)),
        "median_reprojection_error": float(np.median(reprojection_errors)),
        "std_reprojection_error": float(np.std(reprojection_errors)),
        "max_reprojection_error": float(np.max(reprojection_errors)),
        "num_observations": len(reprojection_errors),
        "per_player_stats": per_player_stats,
    }


def intersection_over_plane(o, d):
    """
    args:
        o: (3,) origin of the ray
        d: (3,) direction of the ray

    returns:
        intersection: (3,) intersection point
    """
    # ray is p(t) = o + t * d with scalar param t
    # solve the x and y where z = 0
    t = -o[2] / d[2]
    return o + t * d


def estimate_depth_from_height(skel_2d, K, nose_to_ankle_m=1.65):
    """Estimate the camera-space depth of a person using a nose-to-ankle height prior.

    Uses chained segment lengths (ankle→knee→hip→shoulder→nose) rather than the
    straight-line distance, which is more robust when the person is leaning.

    The pinhole model gives:  pixel_height = focal * real_height / depth
    Rearranging:              depth = focal * real_height / pixel_height

    This depth is a property of the whole person (not one specific joint), valid
    when the camera distance >> body height so all joints share roughly the same depth.

    Args:
        skel_2d:         (15, 2) 2D keypoint detections in pixel coordinates.
        K:               (3, 3) camera intrinsic matrix (K[0,0]=fx, K[1,1]=fy).
        nose_to_ankle_m: Expected real-world nose-to-ankle height in metres (default 1.65).

    Returns:
        float target depth in camera space (z along optical axis), or None if key
        joints are missing or the measured pixel height is degenerate.
    """
    # Joint indices (after OPENPOSE_TO_OURS): 0=Nose, 1=RShoulder, 2=LShoulder,
    # 7=RHip, 8=LHip, 9=RKnee, 10=LKnee, 11=RAnkle, 12=LAnkle
    right_chain = [11, 9, 7, 1, 0]  # RAnkle → RKnee → RHip → RShoulder → Nose
    left_chain = [12, 10, 8, 2, 0]  # LAnkle → LKnee → LHip → LShoulder → Nose

    def chain_pixel_length(chain):
        pts = skel_2d[chain]
        if np.isnan(pts).any():
            return None
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    right_len = chain_pixel_length(right_chain)
    left_len = chain_pixel_length(left_chain)

    if right_len is None and left_len is None:
        return None

    valid_lengths = [l for l in [right_len, left_len] if l is not None]
    pixel_height = float(np.mean(valid_lengths))

    if pixel_height <= 1e-3:
        return None

    focal_length = float((K[0, 0] + K[1, 1]) / 2.0)
    return focal_length * nose_to_ankle_m / pixel_height


def ray_from_xy(xy, K, R, t, dist_coeffs=None):
    """
    Compute the ray from the camera center through the image point (x, y),
    correcting for lens distortion using cv2.undistortPoints.

    cv2.undistortPoints iteratively solves the inverse distortion problem, which
    is more accurate than the closed-form approximation — especially near image
    edges where distortion is large and the approximation breaks down.

    Args:
        xy: (2,) array_like containing pixel coordinates [x, y] in the image.
        K: (3, 3) ndarray representing the camera intrinsic matrix.
        R: (3, 3) ndarray representing the camera rotation matrix.
        t: (3,) ndarray representing the camera translation vector.
        dist_coeffs: (N,) distortion coefficients passed to cv2.undistortPoints
                     (k1, k2, p1, p2[, k3[, k4, k5, k6]]). None means no distortion.

    Returns:
        origin: (3,) ndarray representing the camera center in world coordinates.
        direction: (3,) unit ndarray representing the direction of the ray in world coordinates.
    """
    # cv2.undistortPoints expects shape (1, 1, 2) and returns normalised coords
    # (i.e. with K removed and distortion corrected), so P=None, R=None.
    pts = np.array([[[float(xy[0]), float(xy[1])]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, K, dist_coeffs)  # (1, 1, 2) normalised
    x_undist, y_undist = float(undist[0, 0, 0]), float(undist[0, 0, 1])

    # Construct the undistorted direction in camera coordinates (z = 1).
    d_cam = np.array([x_undist, y_undist, 1.0])

    # Transform the direction to world coordinates.
    direction = R.T @ d_cam
    direction = direction / np.linalg.norm(direction)

    # The camera center in world coordinates is given by -R^T t.
    origin = -R.T @ t
    return origin, direction


def project_points_th(obj_pts, R, C, K, k):
    """Projects 3D points onto 2D image plane using camera intrinsics and distortion.

    args:
        obj_pts: (N, 3) - 3D points in world space
        R: (3, 3) - Rotation matrix
        C: (3,) - Camera center
        K: (3, 3) - Camera intrinsic matrix
        k: (5,) - Distortion coefficients

    returns:
        img_pts: (N, 2) - Projected 2D points
    """

    # Transform world points to camera coordinates
    pts_c = (R @ ((obj_pts - C).unsqueeze(-1))).squeeze(-1)

    # Normalize to get image plane coordinates
    img_pts = pts_c[:, :2] / pts_c[:, 2:]

    # Compute radial distortion
    r2 = (img_pts**2).sum(dim=-1, keepdim=True)

    # clamp to avoid, e.g., too large or negative k1/k2 causing nonsensical distortions
    r2 = torch.clamp(r2, 0, 0.5)

    # polynomial distortion model: x_distorted = x * (1 + k1*r^2 + k2*r^4)
    p = torch.arange(1, k.shape[-1] + 1, device=k.device)
    img_pts = img_pts * (torch.ones_like(r2) + (k * r2.pow(p)).sum(-1, keepdim=True))

    # Apply intrinsics K
    img_pts_h = torch.cat([img_pts, torch.ones_like(img_pts[:, :1])], dim=-1)  # Homogeneous coords
    img_pts = (K @ img_pts_h.unsqueeze(-1)).squeeze(-1)[:, :2]  # Convert back to 2D

    return img_pts


def minimize_reprojection_error(pts_3d, pts_2d, R, C, K, k, iterations=10):
    """
    Optimize 3D points to minimize error when reprojecting them back to 2D.

    args:
        pts_3d: (N, 3)  - Initial 3D points (learnable)
        pts_2d: (N, 2)  - Corresponding 2D points
        R: (N, 3, 3)    - Rotation matrix (fixed)
        C: (N, 3)       - Camera center (fixed)
        K: (N, 3, 3)    - Camera intrinsic matrix (fixed)
        k: (N, 2,)      - Distortion coefficients (fixed)
        iterations: int - Number of optimization steps

    returns:
        t: (N, 3) - Optimized translation
    """
    # Convert 3D points to learnable parameters
    # pts_3d = torch.nn.Parameter(pts_3d.clone().detach().requires_grad_(True))
    t = torch.nn.Parameter(torch.zeros_like(pts_3d).clone().detach().requires_grad_(True))
    offset = torch.tensor([3, 3, 0.2], dtype=pts_3d.dtype, device=pts_3d.device)
    lower_bounds = t - offset
    upper_bounds = t + offset

    # check if there are any NaN values
    assert not torch.isnan(pts_3d).any()
    assert not torch.isnan(pts_2d).any()

    def closure():
        optimizer.zero_grad()
        projected_pts = project_points_th(pts_3d + t, R, C, K, k)
        loss = torch.nn.functional.mse_loss(projected_pts, pts_2d)
        loss.backward()
        return loss

    # iteratively update translation to minimize error between 2D points and reprojected 3D points
    optimizer = optim.LBFGS([t], line_search_fn="strong_wolfe")
    for _ in range(iterations):
        optimizer.step(closure)
        with torch.no_grad():
            t.copy_(torch.clamp(t, lower_bounds, upper_bounds))

    return t.detach()


def fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes, device="cpu"):
    """wrapper function to fine-tune the translation of the 3D predictions to minimize reprojection error

    Uses all 15 joints to compute the reprojection error, providing richer signal for pose refinement.
    Optimizes a single translation per (person, frame) pair across all joints.
    """
    NUM_PERSONS = predictions.shape[0]

    # all 15 joints
    all_joints_3d = predictions  # (NUM_PERSONS, NUM_FRAMES, 15, 3)
    all_joints_2d = skels_2d.transpose(1, 0, 2, 3)  # transpose to (NUM_PERSONS, NUM_FRAMES, 15, 2)

    # from rotation matrices and translations, get all camera centers in world coordinates
    R = np.array([k[0] for k in Rt])
    t = np.array([k[1] for k in Rt])
    C = (-t[:, None] @ R).squeeze(1)

    # replicate camera parameters for each person
    camera_params = {
        "K": cameras["K"][None].repeat(NUM_PERSONS, axis=0),
        "R": R[None].repeat(NUM_PERSONS, axis=0),
        "C": C[None].repeat(NUM_PERSONS, axis=0),
        "k": cameras["k"][None, ..., :2].repeat(NUM_PERSONS, axis=0),
    }

    # for each person, know which frames are valid based on whether the bounding box is NaN or not
    valid = ~np.isnan(boxes).any(axis=-1).transpose(1, 0)  # (NUM_PERSONS, NUM_FRAMES)

    # Extract valid (person, frame) pairs for all 15 joints
    # Shape: (num_valid, 15, 3) and (num_valid, 15, 2)
    all_joints_3d_valid = all_joints_3d[valid]
    all_joints_2d_valid = all_joints_2d[valid]

    # Flatten joints into batch dimension: (num_valid * 15, 3) and (num_valid * 15, 2)
    # This provides 15× richer gradient signal for pose refinement
    pts_3d_flat = all_joints_3d_valid.reshape(-1, 3)
    pts_2d_flat = all_joints_2d_valid.reshape(-1, 2)

    # Replicate camera parameters for each of the 15 joints
    R_valid = camera_params["R"][valid]  # (num_valid, 3, 3)
    C_valid = camera_params["C"][valid]  # (num_valid, 3)
    K_valid = camera_params["K"][valid]  # (num_valid, 3, 3)
    k_valid = camera_params["k"][valid]  # (num_valid, 2)

    R_repeat = np.repeat(R_valid, 15, axis=0)  # (num_valid * 15, 3, 3)
    C_repeat = np.repeat(C_valid, 15, axis=0)  # (num_valid * 15, 3)
    K_repeat = np.repeat(K_valid, 15, axis=0)  # (num_valid * 15, 3, 3)
    k_repeat = np.repeat(k_valid, 15, axis=0)  # (num_valid * 15, 2)

    # iteratively update translation to minimize error across all 15 joints
    traj_3d_flat = minimize_reprojection_error(
        pts_3d=torch.tensor(pts_3d_flat, dtype=torch.float32, device=device),
        pts_2d=torch.tensor(pts_2d_flat, dtype=torch.float32, device=device),
        R=torch.tensor(R_repeat, dtype=torch.float32, device=device),
        C=torch.tensor(C_repeat, dtype=torch.float32, device=device),
        K=torch.tensor(K_repeat, dtype=torch.float32, device=device),
        k=torch.tensor(k_repeat, dtype=torch.float32, device=device),
    )

    # Average translations across 15 joints to get one translation per (person, frame) pair
    # traj_3d_flat shape: (num_valid * 15, 3)
    # Reshape to (num_valid, 15, 3) and take mean along joint dimension
    traj_3d = traj_3d_flat.reshape(-1, 15, 3).mean(dim=1)  # (num_valid, 3)

    return traj_3d, valid


def bundle_adjustment_camera(
    pts_3d, cameras, Rt, dist_maps, iterations=10, window_size=20, device="cpu"
):
    """Refine the camera parameters based on the lane line mask and distance map.

    Adapted from CameraTracker._refine_rotation_with_mask to work across all frames
    and optimize both rotation and translation parameters. Processes frames in sliding windows
    to keep computation tractable while still optimizing all valid frames.

    Args:
        pts_3d: (M, 3) pitch points
        cameras: dict with K, k for all frames
        Rt: list of (R, t) tuples for all frames
        dist_maps: list of distance maps (or None) for each frame
        iterations: max iterations per window
        window_size: number of frames per optimization window (default 100)
        device: device to use
    """
    # N: number of frames, M: number of 3D points

    # Get full camera parameters
    R_full = np.array([k[0] for k in Rt])  # (N, 3, 3)
    t_full = np.array([k[1] for k in Rt])  # (N, 3)

    # Get valid frame indices (frames that have distance maps)
    valid_frame_indices = np.array([i for i, d in enumerate(dist_maps) if d is not None])

    if len(valid_frame_indices) == 0:
        # No valid distance maps, return original parameters
        return [(r_ref, t_ref) for r_ref, t_ref in zip(R_full, t_full, strict=True)]

    # Process valid frames in windows
    R_optimized = R_full.copy()
    t_optimized = t_full.copy()

    for window_start in tqdm(
        range(0, len(valid_frame_indices), window_size), desc="Bundle Adjustment"
    ):
        window_end = min(window_start + window_size, len(valid_frame_indices))
        window_valid_indices = valid_frame_indices[window_start:window_end]

        # Get camera params and distance maps for this window
        R_window = R_full[window_valid_indices].copy()  # (W, 3, 3)
        t_window = t_full[window_valid_indices].copy()  # (W, 3)
        dmaps_window = np.array([dist_maps[i] for i in window_valid_indices])  # (W, H, W_pix)

        # Flatten parameters for optimization
        r_flat = R_window.reshape(-1, 9)  # (W, 9)
        params_init = np.concatenate([r_flat, t_window], axis=1).flatten()  # (W * 12,)

        def objective_function(params):
            # Reshape parameters back
            params_reshaped = params.reshape(-1, 12)  # (W, 12)
            R_params = params_reshaped[:, :9].reshape(-1, 3, 3)  # (W, 3, 3)
            t_params = params_reshaped[:, 9:]  # (W, 3)

            residuals = []

            for i, frame_idx in enumerate(window_valid_indices):
                R = R_params[i]
                t = t_params[i]
                dmap = dmaps_window[i]
                H, W_pix = dmap.shape

                # Enforce rotation matrix orthogonality
                try:
                    R = CameraTracker.find_closest_orthogonal_matrix(R)
                except np.linalg.LinAlgError:
                    # SVD failed, skip orthogonality enforcement for this frame
                    pass

                # Project 3D points
                pts_2d, _ = cv2.projectPoints(
                    pts_3d, cv2.Rodrigues(R)[0], t, cameras["K"][frame_idx], cameras["k"][frame_idx]
                )
                pts_2d = pts_2d.squeeze(axis=1)

                # Round to pixel coordinates, handling NaN/inf gracefully
                xs = np.full(len(pts_2d), -1, dtype=np.int32)
                ys = np.full(len(pts_2d), -1, dtype=np.int32)
                valid_proj = np.isfinite(pts_2d[:, 0]) & np.isfinite(pts_2d[:, 1])
                xs[valid_proj] = np.round(pts_2d[valid_proj, 0]).astype(np.int32)
                ys[valid_proj] = np.round(pts_2d[valid_proj, 1]).astype(np.int32)

                valid_mask = (xs >= 0) & (xs < W_pix) & (ys >= 0) & (ys < H)
                if np.any(valid_mask):
                    xs_valid = xs[valid_mask]
                    ys_valid = ys[valid_mask]
                    frame_loss = dmap[ys_valid, xs_valid].mean()
                else:
                    # If no visible points, penalize lightly
                    frame_loss = np.sqrt(H**2 + W_pix**2)

                residuals.append(frame_loss)

            return np.array(residuals)

        # Set up bounds
        bounds = []
        for _ in range(len(window_valid_indices)):
            bounds.extend([(-1.0, 1.0)] * 9)  # Rotation perturbation
            bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0)])  # Translation perturbation

        params_delta = np.zeros_like(params_init)

        # Optimize this window
        result = scipy.optimize.least_squares(
            lambda delta: objective_function(params_init + delta),
            params_delta,
            method="dogbox",  # More robust for ill-conditioned problems
            bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
            max_nfev=iterations * 50,  # Reduce to avoid numerical issues
        )

        # Extract optimized parameters for this window
        params_opt = params_init + result.x
        params_reshaped = params_opt.reshape(-1, 12)
        R_opt = params_reshaped[:, :9].reshape(-1, 3, 3)
        t_opt = params_reshaped[:, 9:]

        # Enforce final orthogonality
        for i in range(R_opt.shape[0]):
            R_opt[i] = CameraTracker.find_closest_orthogonal_matrix(R_opt[i])

        # Store optimized parameters back into full arrays
        R_optimized[window_valid_indices] = R_opt
        t_optimized[window_valid_indices] = t_opt

    return [(r_ref, t_ref) for r_ref, t_ref in zip(R_optimized, t_optimized, strict=True)]


def process_sequence(
    boxes: np.ndarray,
    cameras: dict,
    skels_3d: np.ndarray,
    skels_2d: np.ndarray,
    video_path: Path | str,
    tracker_options: CameraTrackerOptions,
    device: str = "cpu",
) -> np.ndarray:
    """a naive baseline that uses the bounding boxes to estimate the camera pose
    1. estimate the camera pose using the bounding boxes
    2. periodically refine the camera pose using lane lines
    3. project the 3D skeletons to the 2D image plane and optimize the translation to minimize reprojection error
    """
    NUM_FRAMES, NUM_PERSONS, _ = boxes.shape
    predictions = np.zeros((NUM_PERSONS, NUM_FRAMES, 15, 3))
    predictions.fill(np.nan)
    pitch_points = np.loadtxt("data/pitch_points.txt")

    video = cv2.VideoCapture(video_path)
    camera_tracker = CameraTracker(
        pitch_points=pitch_points,
        fps=video.get(cv2.CAP_PROP_FPS),  # fps = 50
        options=tracker_options,
    )
    camera_tracker.initialize(
        frame_idx=0,
        K=cameras["K"][0],
        k=cameras["k"][0],
        R=cameras["R"][0],
        t=cameras["t"][0],
    )

    Rt = []
    dist_maps = []

    for frame_idx in (pbar := tqdm(range(NUM_FRAMES), desc=f"{video_path.stem}")):
        success, img = video.read()
        if not success:
            print(f"Failed to read frame {frame_idx} from {video_path}")
            break

        # track current camera pose (rotation and translation) based on pitch lines
        # and previously estimated camera pose
        state, dist_map = camera_tracker.track(
            frame_idx=frame_idx,
            frame=img,
            K=cameras["K"][frame_idx],
            dist_coeffs=cameras["k"][frame_idx],
        )

        # if dist_map is None, copy the previous one
        if dist_map is None and len(dist_maps) > 0:
            dist_map = dist_maps[-1]
        dist_maps.append(dist_map)

        yaw, pitch, roll = state.get_ypr()
        pbar.set_postfix_str(f"yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
        Rt.append((state.R.copy(), state.t.copy()))

        # prepare lists to batch-draw visualizations
        frame_skel_2d = []
        frame_skel_3d = []

        # camera parameters for this frame (constant across persons)
        K = cameras["K"][frame_idx]
        k = cameras["k"][frame_idx]
        R, t = Rt[-1]

        # camera center in world coordinates given by -R^T t
        camera_center = -R.T @ t

        for person in range(NUM_PERSONS):
            box = boxes[frame_idx, person]
            # skip if person not visible
            if np.isnan(box).any():
                continue

            skel_2d = skels_2d[frame_idx, person]

            # assumption: lowest point = foot in contact with the ground
            # decide which joint (foot) is in contact with the ground by checking which has
            # largest y (= lowest in image)
            IDX = np.argmax(skel_2d[:, 1])
            x, y = skel_2d[IDX]

            # compute ray from camera center through image point where foot is supposedly touching the ground
            # get origin and normalized direction of the ray in world coordinates
            o, d = ray_from_xy((x, y), K, R, t, k)

            # intersect ray with the ground plane (z=0) to get the 3D location of the foot in world coordinates
            intersection = intersection_over_plane(o, d)

            # convert 3D detections from camera space to world space
            # Q: Do we have to undistort the 3D skeletons?
            # A: Radial distortion is a bending of the picture produced by the lens.
            # If you’re working with image pixels, you must correct for it to know which
            # direction in space they correspond to.
            # But once you’ve moved off the 2‑D picture plane – either by undistorting and
            # back‑projecting the pixel, or because your detector already gives you 3‑D coordinates –
            # the bending effect has no further influence.
            # You’re then just dealing with straight rays and rigid transforms, so you can
            # safely “ignore” distortion from that point onwards.
            skel_3d = skels_3d[frame_idx, person]

            # skel_3d @ R = (R.T @ skel_3d.T).T = (inv(R) @ skel_3d.T).T since R is an orthogonal matrix
            # inverse R to convert from camera space to world space
            raw_skel_world = skel_3d @ R + camera_center

            foot_pos = intersection

            # # Estimate the foot's target depth from the height prior, then walk along the
            # # IDX ray until that camera-space depth is reached.
            # # Both `intersection` and `ray_until_depth` lie on the same ray (o, d), so
            # # their average is also on that ray. Any point on the ray projects back to
            # # the same pixel (x, y), so reprojection of IDX is preserved regardless of
            # # which point we pick. World XY and Z both change along the ray.
            # target_depth = estimate_depth_from_height(skel_2d, K)

            # # how much to trust the height-based depth prior vs the plane intersection
            # lambda_prior = 0.2

            # if target_depth is not None and target_depth > 0:
            #     # Camera-space z of (o + s*d) = R[2]@o + s*(R[2]@d) + t[2].
            #     # Since o = -R.T@t, R[2]@o + t[2] = 0, so s = target_depth / (R[2]@d).
            #     axis_component = float(R[2] @ d)
            #     if abs(axis_component) > 1e-6:
            #         s = target_depth / axis_component
            #         ray_until_depth = o + s * d
            #         foot_pos = (1 - lambda_prior) * intersection + lambda_prior * ray_until_depth
            #     else:
            #         foot_pos = intersection
            # else:
            #     foot_pos = intersection

            skel_3d = raw_skel_world - raw_skel_world[IDX] + foot_pos

            predictions[person, frame_idx] = skel_3d

            # collect for later drawing
            frame_skel_2d.append(skel_2d)
            frame_skel_3d.append(skel_3d)

        # after processing all persons, draw once if debugging enabled
        if "projection" in tracker_options.debug_stages:
            for skel2d, skel3d in zip(frame_skel_2d, frame_skel_3d):
                camera_tracker.debug_vis.draw_3d_points(
                    skel2d, color=(255, 0, 255), connectivity=SKELETON_CONNECTIVITY
                )
                camera_tracker.debug_vis.draw_3d_points(
                    skel3d, R, t, K, k, color=(0, 255, 255), connectivity=SKELETON_CONNECTIVITY
                )

            paused = False
            while True:
                cv2.imshow("Visualization", camera_tracker.debug_vis.frame_curr)
                key = cv2.waitKey(1 if not paused else 0)

                if key == ord(" "):
                    paused = not paused
                    print("Paused" if paused else "Playing")
                elif key == ord("q"):
                    exit()
                else:
                    break

    # optional: bundle adjustmend for camera parameters based on lane line distance maps
    # Rt = bundle_adjustment_camera(
    #     pts_3d=pitch_points,
    #     Rt=Rt,
    #     cameras={k: v[: len(Rt)] for k, v in cameras.items()},
    #     dist_maps=dist_maps,
    # )

    # fine-tune the translation to minimize reprojection error
    traj_3d, valid = fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes, device=device)
    predictions[valid] = predictions[valid] + traj_3d.cpu().numpy()[:, None, :]

    # smoothen the person trajectories based on mid hip keypoint to reduce jitter
    for person in range(NUM_PERSONS):
        predictions[person] = smoothen(predictions[person])

    # update the camera parameters for downstream evaluation, export etc.
    cameras["R"] = np.array([k[0] for k in Rt], dtype=np.float32)
    cameras["t"] = np.array([k[1] for k in Rt], dtype=np.float32)
    return predictions.astype(np.float32)


def load_sequences(sequences_file: Path | str) -> list[str]:
    with open(sequences_file) as f:
        sequences = f.read().splitlines()
    sequences = filter(lambda x: not x.startswith("#"), sequences)
    sequences = [s.strip() for s in sequences]
    return sequences


def main(
    sequences: list[str],
    output: Path | str,
    max_refine_interval: int,
    export_camera: bool,
    visualize: bool,
):

    # create output directory if it doesn't exist
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    debug_stages = ["projection", "flow", "mask"] if visualize else []
    if export_camera:
        camera_dir = output.parent / "calibration"
        camera_dir.mkdir(parents=True, exist_ok=True)
    else:
        camera_dir = None

    if torch.cuda.is_available():
        print("Using GPU")
        device = "cuda"
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon")
        device = "mps"
    else:
        print("Using CPU")
        device = "cpu"

    root = Path("data/")
    solutions = {}
    for sequence in sequences:
        seq_output_fp = output.parent / f"{sequence}.npz"
        if seq_output_fp.exists():
            print(f"Skipping {sequence} since output already exists at {seq_output_fp}")
            solutions[sequence] = np.load(seq_output_fp)[sequence]
            continue

        camera = dict(np.load(root / "cameras" / f"{sequence}.npz"))
        skel2d = np.load(root / "skel_2d" / f"{sequence}.npy")
        skel3d = np.load(root / "skel_3d" / f"{sequence}.npy")
        boxes = np.load(root / "boxes" / f"{sequence}.npy")
        video_path = root / "videos" / f"{sequence}.mp4"

        NUM_FRAMES = boxes.shape[0]
        solutions[sequence] = process_sequence(
            cameras=camera,
            boxes=boxes,
            skels_2d=skel2d[:, :, OPENPOSE_TO_OURS],
            skels_3d=skel3d[:, :, OPENPOSE_TO_OURS],
            video_path=video_path,
            tracker_options=CameraTrackerOptions(
                refine_interval=np.clip(NUM_FRAMES // 500, a_min=1, a_max=max_refine_interval),
                debug_stages=tuple(debug_stages),
            ),
            device=device,
        )

        if export_camera:
            camera_path = camera_dir / f"{sequence}.npz"
            np.savez(camera_path, **camera)

        # save each sequence's predictions separately for easier debugging and visualization
        np.savez_compressed(seq_output_fp, **{sequence: solutions[sequence]})

        # Compute reprojection error diagnostics
        Rt_list = [(camera["R"][i], camera["t"][i]) for i in range(len(camera["R"]))]
        reproj_stats = compute_reprojection_error(
            solutions[sequence], skel2d[:, :, OPENPOSE_TO_OURS], camera, Rt_list, boxes
        )
        print(
            f"  Reprojection stats: mean={reproj_stats['mean_reprojection_error']:.2f}px, "
            f"median={reproj_stats['median_reprojection_error']:.2f}px, "
            f"std={reproj_stats['std_reprojection_error']:.2f}px"
        )
        for player_stats in reproj_stats["per_player_stats"]:
            print(
                f"    Player {player_stats['player_id']:02d}: "
                f"mean_err={player_stats['mean_reprojection_error']:.2f}px, "
                f"median_err={player_stats['median_reprojection_error']:.2f}px, "
                f"x={player_stats['mean_x']:.2f}px, "
                f"y={player_stats['mean_y']:.2f}px, "
                f"n={player_stats['num_observations']}"
            )

    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **solutions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences",
        "-s",
        type=str,
        default="data/sequences_full.txt",
        help="Path to the sequences file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="output/submission_full.npz",
        help="Path to the output npz file",
    )
    parser.add_argument(
        "--refine_interval", "-r", type=int, default=1, help="Interval to refine the camera pose"
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Visualize the tracking results"
    )
    parser.add_argument(
        "--export_camera", "-c", action="store_true", help="Export the camera parameters"
    )
    args = parser.parse_args()

    sequences = load_sequences(args.sequences)
    main(sequences, args.output, args.refine_interval, args.export_camera, args.visualize)
