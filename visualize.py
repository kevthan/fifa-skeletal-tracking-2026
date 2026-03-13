import numpy as np
import cv2
from pathlib import Path
import tqdm

# Importing necessary modules from aitviewer
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.viewer import Viewer
from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import OpenCVCamera


def create_billboard(camera, img_folder, distance=200, draw_fn=None):
    """Create a billboard from a sequence of images."""
    img_paths = sorted(img_folder.glob("*.jpg"))
    H, W = camera.rows, camera.cols
    pc = Billboard.from_camera_and_distance(
        camera, distance, W, H, textures=[str(path) for path in img_paths], image_process_fn=draw_fn
    )
    return pc


def convert_video_to_images(video_path, output_folder):
    """Convert video to images."""
    if output_folder.exists():
        print(f"Output folder {output_folder} already exists. Skipping conversion.")
        return

    output_folder.mkdir(exist_ok=True, parents=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    with tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Converting video to images:") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fp = output_folder / f"{frame_id:06d}.jpg"
            if not fp.exists():
                cv2.imwrite(str(fp), frame)
            frame_id += 1
            pbar.update(1)
    cap.release()


class Skel15:
    OPENPOSE_TO_OURS = [0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]

    joint_names = ['Nose', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist',
                   'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RBigToe', 'LBigToe']

    bones = np.array(
        [
            [0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [1, 7], [2, 8],
            [7, 9], [8, 10], [9, 11], [10, 12], [11, 13], [12, 14]
        ]
    )


def make_draw_func(camera, boxes):
    # generate a set of colors for the boxes
    num_players = boxes.shape[1]
    colors = np.random.rand(num_players, 3)
    colors = (colors * 255).astype(np.uint8)

    def _draw_func(img, current_frame_id):
        current_frame_id = min(current_frame_id, len(camera["K"]) - 1)
        for i in range(num_players):
            box = boxes[current_frame_id, i]
            if np.isnan(box).any():
                continue
            box = box.astype(np.int32)
            x1, y1, x2, y2 = box
            # drawing a rectangle around the box
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[i].tolist(), 2)
            # also draw a label with the player id
            cv2.putText(img, f"Player #{i:02d}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i].tolist(), 2)
        return img

    return _draw_func


def generate_player_colors(num_players):
    """Generate distinct colors for each player."""
    rng = np.random.default_rng(42)
    colors = rng.random((num_players, 3))
    # Make colors more vivid by pushing towards saturation
    colors = 0.3 + 0.7 * colors
    return colors


def add_skeleton_renderables(viewer, predictions, num_frames):
    """Add 3D skeleton renderables for each player.

    Args:
        viewer: aitviewer Viewer instance
        predictions: (N_persons, N_frames, 15, 3) array of 3D joint positions
        num_frames: number of frames in the sequence
    """
    num_persons = predictions.shape[0]
    colors = generate_player_colors(num_persons)

    for person_id in range(num_persons):
        joints = predictions[person_id]  # (N_frames, 15, 3)

        # Check if this person has any valid frames
        valid_mask = ~np.isnan(joints).any(axis=(-1, -2))  # (N_frames,)
        if not valid_mask.any():
            continue

        # Fill NaN frames with zeros (they won't be visible but keeps shape consistent)
        joints_filled = joints.copy()
        joints_filled[~valid_mask] = 0.0

        color = (*colors[person_id].tolist(), 1.0)
        skel = Skeletons(
            joint_positions=joints_filled,
            joint_connections=Skel15.bones,
            radius=0.02,
            color=color,
            name=f"Player #{person_id:02d}",
        )
        viewer.scene.add(skel)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="ARG_FRA_183303")
    parser.add_argument("--predictions", "-p", type=Path, default=Path("outputs/submission_full.npz"),
                        help="Path to the output predictions NPZ file")
    parser.add_argument("--calibration_dir", type=Path, default=Path("outputs/calibration/"),
                        help="Directory containing calibrated camera NPZ files (from --export_camera)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output_path", type=Path, default=Path("outputs/result_vis"))
    args = parser.parse_args()

    # define constants
    C.z_up = True

    # Setup viewer and load data
    H, W = 1080, 1920
    if args.headless:
        viewer = HeadlessRenderer(size=(W, H))
    else:
        viewer = Viewer(size=(W, H))
    clip_name = args.sequence
    data_dir = Path("data/")
    video_path = data_dir / f"videos/{clip_name}.mp4"

    # Load calibrated camera (per-frame R, t) from pipeline output
    calibrated_camera_path = args.calibration_dir / f"{clip_name}.npz"
    if not calibrated_camera_path.exists():
        raise FileNotFoundError(
            f"Calibrated camera not found at {calibrated_camera_path}. "
            f"Run main.py with --export_camera first."
        )
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")

    camera_params = dict(np.load(calibrated_camera_path))
    boxes = np.load(data_dir / "boxes" / f"{clip_name}.npy")

    # Load 3D predictions
    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Predictions not found at {args.predictions}. "
            f"Run main.py first to generate predictions."
        )
    all_predictions = dict(np.load(args.predictions))
    if clip_name not in all_predictions:
        raise KeyError(
            f"Sequence '{clip_name}' not found in predictions file. "
            f"Available: {list(all_predictions.keys())}"
        )
    predictions = all_predictions[clip_name]  # (N_persons, N_frames, 15, 3)

    # Setup camera and billboard
    img_folder = data_dir / "images" / clip_name
    convert_video_to_images(video_path, img_folder)

    # Build Rt matrix: per-frame R (N, 3, 3) and t (N, 3) -> Rt (N, 3, 4)
    camera_params["Rt"] = np.concatenate(
        [camera_params["R"], camera_params["t"][..., None]], axis=-1
    )
    camera = OpenCVCamera(camera_params["K"], camera_params["Rt"], W, H, viewer=viewer, name="Overlay")
    billboard = create_billboard(camera, img_folder, 200, make_draw_func(camera_params, boxes))
    viewer.scene.add(billboard)
    viewer.scene.add(camera)

    # Add 3D skeleton predictions
    num_frames = boxes.shape[0]
    add_skeleton_renderables(viewer, predictions, num_frames)

    # Configure lighting
    light = viewer.scene.lights[0]
    light.shadow_enabled = True
    light.azimuth = 270
    light.elevation = 0
    light.shadow_map_size = 64
    light.shadow_map_near = 0.01
    light.shadow_map_far = 50
    viewer.scene.lights[1].shadow_enabled = False

    # Finalize setup and run viewer
    viewer.scene.floor.enabled = False
    viewer.set_temp_camera(camera)
    if args.headless:
        args.output_path.mkdir(exist_ok=True, parents=True)
        viewer.save_video(video_dir=str(args.output_path / f"{clip_name}.mp4"), output_fps=60, ensure_no_overwrite=False)
    else:
        viewer.run()
