"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [1, 1, 0],  # Yellow
    [1, 0, 1],  # Magenta
    [0, 1, 1],  # Cyan
    [0.5, 1, 0.5],  # Gray
    [1, 0.5, 0],  # Orange
    [0.5, 0, 1],  # Purple
    [0, 0.5, 0.5],  # Teal
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba

def filter_points_and_boxes_xy(points, gt_boxes, ref_boxes, x_range, y_range, z_range=None):
    """
    위에서 본 XY 범위로 Point Cloud와 Boxes를 필터링합니다.

    Args:
        points (numpy.ndarray): (N, 3) 또는 (N, 4), Point Cloud 데이터.
        gt_boxes (numpy.ndarray): (M, 10), Ground Truth Bounding Boxes.
        x_range (tuple): XY 평면의 X축 범위 (min, max).
        y_range (tuple): XY 평면의 Y축 범위 (min, max).
        z_range (tuple): 선택적, Z축 범위 (min, max).

    Returns:
        points_filtered (numpy.ndarray): XY 범위 내의 Point Cloud.
        gt_boxes_filtered (numpy.ndarray): XY 범위 내의 Bounding Boxes.
    """
    mask_points = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
                  (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])

    if z_range is not None:
        mask_points = mask_points & (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])

    points_filtered = points[mask_points]


    cx, cy = gt_boxes[:, 0], gt_boxes[:, 1]
    mask_boxes = (cx >= x_range[0]) & (cx <= x_range[1]) & \
                 (cy >= y_range[0]) & (cy <= y_range[1])
    gt_boxes_filtered = gt_boxes[mask_boxes]

    cx, cy = ref_boxes[:, 0], ref_boxes[:, 1]
    mask_boxes = (cx >= x_range[0]) & (cx <= x_range[1]) & \
                 (cy >= y_range[0]) & (cy <= y_range[1])
    ref_boxes_filtered = ref_boxes[mask_boxes]

    return points_filtered, gt_boxes_filtered, ref_boxes_filtered

def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, save_path=None):
    if save_path == None:
        print("Please set save path for vis results")
        assert False
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    points, gt_boxes, ref_boxes = filter_points_and_boxes_xy(points, gt_boxes, ref_boxes, (-50, 50), (-50, 50))

    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=False, width=720, height=720)

    vis.get_render_option().point_size = 0.1
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    view_control = vis.get_view_control()
    # view_control.set_zoom(0.5)
    view_control.set_lookat([0, 0, 10])
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, -1, 0])
    
    vis.capture_screen_image(filename=save_path, do_render=True)
    vis.destroy_window()

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=None, ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
            vis.add_geometry(line_set)
        else:
            line_set.paint_uniform_color(color)
            # num_lines = len(line_set.lines)
            # line_colors = [box_colormap[ref_labels[i] - 1]] * num_lines
            # line_set.colors = open3d.utility.Vector3dVector(line_colors)
            if score[i] > 0.3:
                vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis