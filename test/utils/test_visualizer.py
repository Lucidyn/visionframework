"""
Visualizer 基础测试。

验证：
- Visualizer 初始化
- 各绘制方法返回预期结构
- 可视化输出完整性
"""

import numpy as np

from visionframework import Visualizer, Detection, Track, Pose, KeyPoint


def _make_dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_dummy_detections() -> list:
    return [
        Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"),
        Detection(bbox=(300, 150, 400, 250), confidence=0.85, class_id=1, class_name="car"),
    ]


def _make_dummy_tracks() -> list:
    return [
        Track(track_id=1, bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"),
        Track(track_id=2, bbox=(300, 150, 400, 250), confidence=0.85, class_id=1, class_name="car"),
    ]


def _make_dummy_poses() -> list:
    return [
        Pose(
            bbox=(100, 100, 200, 300),
            keypoints=[
                KeyPoint(keypoint_id=0, keypoint_name="nose",           x=150, y=120, confidence=0.9),
                KeyPoint(keypoint_id=1, keypoint_name="left_eye",       x=130, y=110, confidence=0.8),
                KeyPoint(keypoint_id=2, keypoint_name="right_eye",      x=170, y=110, confidence=0.8),
                KeyPoint(keypoint_id=3, keypoint_name="neck",           x=150, y=180, confidence=0.85),
                KeyPoint(keypoint_id=4, keypoint_name="left_shoulder",  x=120, y=220, confidence=0.8),
                KeyPoint(keypoint_id=5, keypoint_name="right_shoulder", x=180, y=220, confidence=0.8),
            ],
            confidence=0.85,
            pose_id=1,
        )
    ]


def test_visualizer_initialization() -> None:
    """Visualizer 应能用默认或自定义参数成功初始化。"""
    visualizer = Visualizer()
    assert isinstance(visualizer, Visualizer)

    custom_config = {
        "line_thickness": 2,
        "font_scale": 0.5,
        "font_thickness": 1,
        "colors": {0: (0, 255, 0), 1: (0, 0, 255)},
    }
    custom_visualizer = Visualizer(config=custom_config)
    assert isinstance(custom_visualizer, Visualizer)


def test_visualizer_draw_detections() -> None:
    """draw_detections 应返回与输入形状相同的图像数组。"""
    visualizer = Visualizer()
    img = _make_dummy_image()
    detections = _make_dummy_detections()

    vis_image = visualizer.draw_detections(img.copy(), detections)
    assert isinstance(vis_image, np.ndarray)
    assert vis_image.shape == img.shape


def test_visualizer_draw_tracks() -> None:
    """draw_tracks 应返回与输入形状相同的图像数组。"""
    visualizer = Visualizer()
    img = _make_dummy_image()
    tracks = _make_dummy_tracks()

    vis_image = visualizer.draw_tracks(img.copy(), tracks)
    assert isinstance(vis_image, np.ndarray)
    assert vis_image.shape == img.shape


def test_visualizer_draw_poses() -> None:
    """draw_poses 应返回与输入形状相同的图像数组。"""
    visualizer = Visualizer()
    img = _make_dummy_image()
    poses = _make_dummy_poses()

    vis_image = visualizer.draw_poses(img.copy(), poses)
    assert isinstance(vis_image, np.ndarray)
    assert vis_image.shape == img.shape


def test_visualizer_draw_results() -> None:
    """draw_results 应返回绘制了所有结果的图像数组。"""
    visualizer = Visualizer()
    img = _make_dummy_image()
    detections = _make_dummy_detections()
    tracks = _make_dummy_tracks()
    poses = _make_dummy_poses()

    vis_image = visualizer.draw_results(
        img.copy(),
        detections=detections,
        tracks=tracks,
        poses=poses,
    )
    assert isinstance(vis_image, np.ndarray)
    assert vis_image.shape == img.shape


def test_visualizer_with_empty_inputs() -> None:
    """Visualizer 应能优雅处理空输入。"""
    visualizer = Visualizer()
    img = _make_dummy_image()

    for vis_image in [
        visualizer.draw_detections(img.copy(), []),
        visualizer.draw_tracks(img.copy(), []),
        visualizer.draw_poses(img.copy(), []),
        visualizer.draw_results(img.copy(), detections=[], tracks=[], poses=[]),
    ]:
        assert isinstance(vis_image, np.ndarray)
        assert vis_image.shape == img.shape
