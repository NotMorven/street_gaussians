import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class TrajectoryEditor(QMainWindow):
    """
    to edit track_info.txt, implementated by PyQt5 and matplotlib.
    """
    def __init__(self):
        super().__init__()
        self.tracks = {}  # 存储所有轨迹，key为track_id
        self.ego_poses = {}  # 存储ego poses
        self.current_track_id = None
        self.selected_point = None
        self.data_dir = None  # 存储数据目录路径
        
        # bezier smoothing参数
        self.window_size = 15  # 前后影响的轨迹点数量
        self.bezier_points = None  # 存储当前编辑的贝塞尔控制点
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Trajectory Editor')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QVBoxLayout()
        
        # 轨迹文件加载按钮
        load_btn = QPushButton('Load Trajectory File')
        load_btn.clicked.connect(self.load_trajectory)
        
        # 添加显示ego轨迹的复选框
        self.ego_btn = QPushButton('Show Ego Trajectory')
        self.ego_btn.clicked.connect(self.plot_ego_only)
        
        # 显示所有轨迹按钮
        show_all_btn = QPushButton('Show All Trajectories')
        show_all_btn.clicked.connect(self.show_all_trajectories)
        
        # 添加清除画布按钮
        clear_btn = QPushButton('Clear Canvas')
        clear_btn.clicked.connect(self.clear_canvas)
        
        # Track ID选择下拉框
        self.track_selector = QComboBox()
        self.track_selector.currentIndexChanged.connect(self.change_track)
        
        # 保存按钮
        save_btn = QPushButton('Save Trajectory')
        save_btn.clicked.connect(self.save_trajectory)
        
        control_panel.addWidget(load_btn)
        control_panel.addWidget(self.ego_btn)  # 添加ego轨迹复选框
        control_panel.addWidget(show_all_btn)  # 添加新按钮
        control_panel.addWidget(clear_btn)  # 添加清除画布按钮
        control_panel.addWidget(QLabel('Select Track ID:'))
        control_panel.addWidget(self.track_selector)
        control_panel.addWidget(save_btn)
        control_panel.addStretch()
        
        # 右侧matplotlib画布
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
        # 添加到主布局
        layout.addLayout(control_panel, stretch=1)
        layout.addWidget(self.canvas, stretch=4)
        main_widget.setLayout(layout)

    def load_trajectory(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Load Trajectory File', '', 'Text Files (*.txt)')
        if filename:
            # 获取数据目录路径
            self.data_dir = str(Path(filename).parent.parent)
            self.load_ego_poses()
            
            self.tracks.clear()
            with open(filename, 'r') as f:
                self.header = f.readline()  # 保存header
                for line in f:
                    data = line.strip().split()
                    frame_id = int(data[0])
                    track_id = int(data[1])
                    object_class = str(data[2])
                    if object_class != 'vehicle':
                        continue
                        
                    # 获取ego pose
                    ego_pose = self.ego_poses[frame_id]
                    
                    # 将相对坐标转换为绝对坐标
                    rel_x, rel_y, rel_z = float(data[7]), float(data[8]), float(data[9])
                    rel_pos = np.array([rel_x, rel_y, rel_z, 1.0])
                    abs_pos = (ego_pose + rel_pos)[:3]
                    # abs_pos_homo = ego_pose @ rel_pos
                    # abs_pos = abs_pos_homo[:3]
                    
                    # 转换朝向角
                    rel_heading = float(data[10])
                    # abs_heading = rel_heading + np.arctan2(ego_pose[1,0], ego_pose[0,0])
                    abs_heading = rel_heading
                    
                    if track_id not in self.tracks:
                        self.tracks[track_id] = []
                    
                    self.tracks[track_id].append({
                        'frame_id': frame_id,
                        'track_id': track_id,
                        'x': abs_pos[0],  # 存储绝对坐标
                        'y': abs_pos[1],
                        'z': abs_pos[2],
                        # 'heading': rel_heading, 
                        'heading': abs_heading,
                        'original_line': line
                    })
            
            # 更新Track ID选择器
            self.track_selector.clear()
            self.track_selector.addItems([str(tid) for tid in sorted(self.tracks.keys())])
            if self.tracks:
                self.current_track_id = list(self.tracks.keys())[0]
                self.plot_trajectory()

    def load_ego_poses(self):
        ego_pose_dir = os.path.join(self.data_dir, 'ego_pose')
        
        self.ego_trajectory = []
        
        for pose_file in sorted(os.listdir(ego_pose_dir)):
            if pose_file.endswith('.txt'):
                frame_id = int(pose_file.split('.')[0])
                pose_path = os.path.join(ego_pose_dir, pose_file)
                ego_pose = np.loadtxt(pose_path)
                ego_position = ego_pose[:3, 3]
                ego_heading = np.arctan2(ego_pose[1, 0], ego_pose[0, 0])  # TODO: not right

                # self.ego_poses[frame_id] = ego_position
                self.ego_poses[frame_id] = ego_pose
                self.ego_trajectory.append({
                    'frame_id': frame_id,
                    'x': ego_position[0],
                    'y': ego_position[1],
                    'z': ego_position[2],
                    'heading': ego_heading
                })

    def plot_ego_only(self):
        self.ax.clear()
        if self.ego_trajectory:
            x = [p['x'] for p in self.ego_trajectory]
            y = [p['y'] for p in self.ego_trajectory]
            self.ax.plot(x, y, 'g-', linewidth=2, label='Ego Trajectory')
            self.ax.scatter(x[0], y[0], c='g', marker='^', s=100, label='Ego Start')
            self.ax.scatter(x[-1], y[-1], c='g', marker='s', s=100, label='Ego End')
            self.ax.legend()
            self.ax.grid(True)
            self.ax.set_title('Ego Trajectory')
        self.canvas.draw()

    def change_track(self, index):
        if index >= 0:
            self.current_track_id = int(self.track_selector.currentText())
            self.plot_trajectory()

    def plot_trajectory(self):
        if self.current_track_id is None:
            return
        
        self.ax.clear()

        # 先绘制ego轨迹，使用较浅的颜色和较细的线条表示不可编辑
        if self.ego_trajectory:
            ego_x = [p['x'] for p in self.ego_trajectory]
            ego_y = [p['y'] for p in self.ego_trajectory]
            self.ax.plot(ego_x, ego_y, color='lightgray', linewidth=1, label='Ego (Non-editable)', zorder=1)
            self.ax.scatter(ego_x[0], ego_y[0], c='lightgray', marker='^', s=50, label='Ego Start')
            self.ax.scatter(ego_x[-1], ego_y[-1], c='lightgray', marker='s', s=50, label='Ego End')

        # 绘制当前选中的轨迹，使用明显的颜色表示可编辑
        track = self.tracks[self.current_track_id]
        x = [p['x'] for p in track]
        y = [p['y'] for p in track]
        
        self.ax.plot(x, y, 'b-', linewidth=2, label='Selected Track')
        self.ax.scatter(x, y, c='r', s=50, picker=5, label='Editable Points')
        self.ax.legend()
        self.ax.set_title(f'Track ID: {self.current_track_id}')
        self.ax.grid(True)
        self.canvas.draw()

    # 添加新的显示所有轨迹的方法
    def show_all_trajectories(self):
        if not self.tracks:
            return
        
        self.ax.clear()
        
        # 先绘制ego轨迹
        if self.ego_trajectory:
            ego_x = [p['x'] for p in self.ego_trajectory]
            ego_y = [p['y'] for p in self.ego_trajectory]
            self.ax.plot(ego_x, ego_y, 'k-', linewidth=2, label='Ego', zorder=1)
            self.ax.scatter(ego_x[0], ego_y[0], c='k', marker='^', s=100, label='Ego Start')
            self.ax.scatter(ego_x[-1], ego_y[-1], c='k', marker='s', s=100, label='Ego End')
        
        # 使用不同颜色显示不同轨迹
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.tracks)))
        
        for (track_id, track), color in zip(self.tracks.items(), colors):
            x = [p['x'] for p in track]
            y = [p['y'] for p in track]
            
            # 绘制轨迹线和散点
            self.ax.plot(x, y, '-', color=color, label=f'Track {track_id}')
            self.ax.scatter(x, y, c=[color], s=30, alpha=0.6)
            
            # 添加起点和终点标记
            self.ax.scatter(x[0], y[0], c=[color], marker='^', s=100, label=f'Start {track_id}')
            self.ax.scatter(x[-1], y[-1], c=[color], marker='s', s=100, label=f'End {track_id}')
        
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.set_title('All Trajectories')
        self.ax.grid(True)
        
        # 调整图形布局以显示图例
        self.figure.tight_layout()
        self.canvas.draw()
    
    # 添加清除画布方法
    def clear_canvas(self):
        self.ax.clear()
        self.ax.set_title('Canvas Cleared')
        self.ax.grid(True)
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        track = self.tracks[self.current_track_id]
        x = [p['x'] for p in track]
        y = [p['y'] for p in track]
        
        distances = [(abs(event.xdata - px) + abs(event.ydata - py), i) 
                    for i, (px, py) in enumerate(zip(x, y))]
        closest_dist, closest_idx = min(distances)
        
        if closest_dist < 1.0:  # 设置选择阈值
            self.selected_point = closest_idx
            self.original_pos = (x[closest_idx], y[closest_idx])

    def bezier_curve(self, points, num_points=50):
        """生成贝塞尔曲线"""
        n = len(points) - 1
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))
        
        for i in range(n + 1):
            curve += np.outer(comb(n, i) * (t ** i) * ((1 - t) ** (n - i)), points[i])
            
        return curve
    
    def calculate_influence_weights(self, dist, window_size):
        """计算影响权重，使用高斯衰减"""
        sigma = window_size / 3.0
        weights = np.exp(-0.5 * (dist / sigma) ** 2)
        return weights
        
    def update_trajectory_smooth(self, event):
        """使用贝塞尔曲线更新轨迹点"""
        if self.selected_point is None:
            return
            
        track = self.tracks[self.current_track_id]
        n_points = len(track)
        
        # 计算影响范围
        start_idx = max(0, self.selected_point - self.window_size)
        end_idx = min(n_points, self.selected_point + self.window_size + 1)
        
        # 获取受影响的点
        affected_points = track[start_idx:end_idx]
        original_positions = np.array([[p['x'], p['y']] for p in affected_points])
        
        # 计算移动量
        delta_x = event.xdata - track[self.selected_point]['x']
        delta_y = event.ydata - track[self.selected_point]['y']
        
        # 计算影响权重
        distances = np.abs(np.arange(start_idx, end_idx) - self.selected_point)
        weights = self.calculate_influence_weights(distances, self.window_size)
        
        # 更新位置
        for i, (point, weight) in enumerate(zip(affected_points, weights)):
            point['x'] += delta_x * weight
            point['y'] += delta_y * weight
        
        # 使用贝塞尔曲线平滑轨迹
        control_points = np.array([[p['x'], p['y']] for p in affected_points])
        smooth_curve = self.bezier_curve(control_points)
        
        # 更新轨迹点位置
        for i, point in enumerate(affected_points):
            interp_idx = int(i * (len(smooth_curve) - 1) / (len(affected_points) - 1))
            point['x'] = smooth_curve[interp_idx, 0]
            point['y'] = smooth_curve[interp_idx, 1]
        
        self.plot_trajectory()

    def on_motion(self, event):
        if event.inaxes != self.ax or self.selected_point is None:
            return
        
        # track = self.tracks[self.current_track_id]
        # track[self.selected_point]['x'] = event.xdata
        # track[self.selected_point]['y'] = event.ydata
        # self.plot_trajectory()
        self.update_trajectory_smooth(event)

    def on_release(self, event):
        self.selected_point = None

    def save_trajectory(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Trajectory File', '', 'Text Files (*.txt)')
        if filename:
            with open(filename, 'w') as f:
                f.write(self.header)
                for track_id, track in self.tracks.items():
                    for point in track:
                        frame_id = point['frame_id']
                        ego_pose = self.ego_poses[frame_id]
                        # ego_pose_inv = np.linalg.inv(ego_pose)
                        
                        # 将绝对坐标转换回相对坐标
                        abs_pos = np.array([point['x'], point['y'], point['z'], 1.0])
                        # rel_pos = ego_pose_inv @ abs_pos
                        rel_pos = abs_pos - ego_pose[..., 3]
                        
                        # 转换朝向角
                        abs_heading = point['heading']
                        # rel_heading = abs_heading - np.arctan2(ego_pose[1,0], ego_pose[0,0])
                        rel_heading = abs_heading
                        
                        # 更新原始行中的坐标和朝向
                        parts = point['original_line'].split()
                        parts[7] = f"{rel_pos[0]:.6f}"
                        parts[8] = f"{rel_pos[1]:.6f}"
                        parts[9] = f"{rel_pos[2]:.6f}"
                        parts[10] = f"{rel_heading:.6f}"
                        # parts[10] = abs_heading
                        f.write(' '.join(parts) + '\n')

class csv2trajInfo():
    def __init__(self, csv_path, 
                 track_info_file=None, 
                 ego_pose_dir=None):
        """
        process cvs_data from scene_dig, and rewrite track_info.txt
        """
        self.csv_path = csv_path
        self.origin_traj_data, self.csv_tracks = self.load_csv()
        
        self.track_info_file = track_info_file
        self.data_dir = ego_pose_dir
        self.tracks = {}  # 存储所有轨迹，key为track_id
        self.ego_poses = {}  # 存储ego poses
        self.ego_trajectory = []
        self.load_ego_poses()
        self.load_track_info()
        
    def load_csv(self):
        """
        Load and process CSV file containing vehicle trajectory information.
        Dynamically determines the number of vehicles from CSV headers.
        Returns processed trajectory information as a list of dictionaries.
        """
        import pandas as pd
        import numpy as np
        
        # Read the CSV file
        df = pd.read_csv(self.csv_path)
        
        # Get all column names and find vehicle columns
        columns = df.columns.tolist()
        id_columns = [col for col in columns if col.endswith('_x') and col.startswith('car')]
        num_vehicles = len(id_columns)
        # print(columns)
        # print(id_columns)
        # print(f"Number of vehicles: {num_vehicles}")
        
        # get number of rows
        num_rows = df.shape[0]
        print(f"file: {self.csv_path}\nNumber of rows: {num_rows}")
        
        # Create empty lists to store reorganized data
        origin_data = {}
        """data structure
        {
            'id': {
                'time': time,
                'car0_id': {
                    'id_value': id_value,
                    'x': x,
                    'y': y,
                    'v': v,
                    'heading': heading
                }, 
                'car1_id': {
                    'id_value': id_value,
                    'x': x,
                    'y': y,
                    'v': v,
                    'heading': heading
                }, 
                ...
            }
            ...
        }
        """
        # Iterate through each row and extract vehicle information
        for idx, row in df.iterrows():
            cur_data_itm = {
                'time': row['time']
            }
            # time = row['time']
            
            # Process each vehicle's data
            for i in range(num_vehicles):
                trail = row[f'car{i}_x']
                # Skip if car_id is NaN (no vehicle data)
                if pd.isna(trail):
                    continue
                
                # car_id = row[f'car{i}_id']
                cur_car_data = {
                    # 'id_value': int(car_id),
                    'x': row[f'car{i}_x'],
                    'y': row[f'car{i}_y'],
                    'v': row[f'car{i}_v'],
                    'heading': row[f'car{i}_heading']
                }
                cur_data_itm[f'car{i}_id'] = cur_car_data
            # Append data
            origin_data[idx] = cur_data_itm
        
        # visualize origin_data
        # for key, value in origin_data.items():
        #     print(f"Key: {key}")
        #     for k, v in value.items():
        #         print(f"===> {k}: {v}")
        # exit(0)
        
        proc_tracks = {}
        """ tracks structure
        {
            'car0_id': [
                {
                    'frame_id': int, 
                    'time_stamp': float,
                    'car_id': int,
                    'rel_x': float,
                    'rel_y': float,
                    'rel_z': float,
                    'abs_x': float,
                    'abs_y': float,
                    'abs_z': float,
                    'heading': float,
                    'speed': float,
                }, 
                ...
            ], 
            'car1_id': [
                ...
            ], 
            ...
        }
        """
        #### process data to coordinate according to ego
        for idx, row in df.iterrows():
            frame_id = idx
            time_stamp = row['time']
            
            for i in range(num_vehicles):
                trail = row[f'car{i}_x']
                # Skip if car_id is NaN (no vehicle data)
                if pd.isna(trail):
                    continue
                
                if f'car{i}_id' not in proc_tracks:
                    proc_tracks[f'car{i}_id'] = []
                
                rel_x, rel_y, rel_z = 0.0, 0.0, 0.0
                # car0 is ego
                if i != 0:
                    rel_x = row[f'car{i}_x'] - row['car0_x']
                    rel_y = (row[f'car{i}_y'] - row['car0_y'])
                    rel_z = 0.0
                
                proc_tracks[f'car{i}_id'].append({
                    'frame_id': frame_id, 
                    'time_stamp': time_stamp,
                    # 'car_id': int(car_id),
                    'rel_x': rel_x,
                    'rel_y': rel_y,
                    'rel_z': 0.0,
                    'abs_x': row[f'car{i}_x'],
                    'abs_y': row[f'car{i}_y'],
                    'abs_z': 0.0,
                    'heading': -row[f'car{i}_heading'],  # TODO giving a '-' for now
                    'speed': row[f'car{i}_v'],
                })
        # # visualize tracks
        # for value in proc_tracks['car1_id']:
        #     print(value)
        # exit(0)
        
        return origin_data, proc_tracks
    
    def load_ego_poses(self):
        ego_pose_dir = os.path.join(self.data_dir)
        
        self.ego_trajectory = []
        
        for pose_file in sorted(os.listdir(ego_pose_dir)):
            if pose_file.endswith('.txt'):
                frame_id = int(pose_file.split('.')[0])
                pose_path = os.path.join(ego_pose_dir, pose_file)
                ego_pose = np.loadtxt(pose_path)
                ego_position = ego_pose[:3, 3]
                ego_heading = np.arctan2(ego_pose[1, 0], ego_pose[0, 0]) # TODO: not right

                # self.ego_poses[frame_id] = ego_position
                self.ego_poses[frame_id] = ego_pose
                self.ego_trajectory.append({
                    'frame_id': frame_id,
                    'x': ego_position[0],
                    'y': ego_position[1],
                    'z': ego_position[2],
                    'heading': ego_heading
                })
    
    def load_track_info(self):
        self.tracks.clear()
        
        with open(self.track_info_file, 'r') as f:
            self.header = f.readline()  # 保存header
            for line in f:
                data = line.strip().split()
                
                frame_id = int(data[0])
                track_id = int(data[1])
                object_class = str(data[2])
                if object_class != 'vehicle':
                    continue
                rel_x, rel_y, rel_z = float(data[7]), float(data[8]), float(data[9])
                rel_heading = float(data[10])
                speed = float(data[11])
                
                # 将相对坐标转换为绝对坐标
                def rel2abs(rel_x, rel_y, rel_z, rel_heading):                    
                    # 获取ego pose
                    # ego_pose = self.ego_poses[frame_id]  # RT matrix
                    ego_pose = self.ego_poses[frame_id][..., 3]  # position
                    ego_heading = np.arctan2(self.ego_poses[frame_id][1, 0], self.ego_poses[frame_id][0, 0])
                    
                    # position
                    rel_pose = np.array([rel_x, rel_y, rel_z, 1.0])
                    # abs_pose_homo = ego_pose @ rel_pose
                    # abs_pose = abs_pose_homo[:3]
                    abs_pose = (ego_pose + rel_pose)[:3]
                    # heading
                    abs_heading = rel_heading
                    return abs_pose, abs_heading
                
                abs_pose, abs_heading = rel2abs(rel_x, rel_y, rel_z, rel_heading)
                
                if track_id not in self.tracks:
                    self.tracks[track_id] = []
                
                self.tracks[track_id].append({
                    'frame_id': frame_id, 
                    'track_id': track_id, 
                    'rel_x': rel_x, 
                    'rel_y': rel_y, 
                    'rel_z': rel_z, 
                    'abs_x': abs_pose[0], 
                    'abs_y': abs_pose[1], 
                    'abs_z': abs_pose[2], 
                    'heading': abs_heading, 
                    'speed': speed,
                    'original_line': line
                })

    def edit_track_info(self, save_name):
        csv_freq = 20
        target_tracks_freq = 10
        # 选择要编辑的原始轨迹
        csv_id = "car1_id"
        track_id = 4
        
        track = self.tracks[track_id]
        csv_track = self.csv_tracks[csv_id]
        # 选择要编辑的轨迹点
        traj_start = 0
        traj_end = 100
        for i in range(traj_start, traj_end+1):
            self.tracks[track_id][i+98]['rel_x'] = csv_track[i*2]['rel_x']+8
            self.tracks[track_id][i+98]['rel_y'] = csv_track[i*2]['rel_y']
            # self.tracks[track_id][i+98]['rel_z'] = csv_track[i*2]['rel_z']
            self.tracks[track_id][i+98]['heading'] = csv_track[i*2]['heading']
        
        # # save
        # save_name = r"track_info_edited.txt"
        full_path = os.path.join(
            "/home/wicv/workspace/jiangchengzhi/street_gaussians/data/waymo/training/002/track", 
            save_name
        )
        with open(full_path, 'w') as f:
            f.write(self.header)
            for track_id, track in self.tracks.items():
                for point in track:
                    frame_id = point['frame_id']
                    parts = point['original_line'].split()
                    parts[7] = f"{point['rel_x']}"
                    parts[8] = f"{point['rel_y']}"
                    parts[9] = f"{point['rel_z']}"
                    parts[10] = f"{point['heading']}"
                    f.write(' '.join(parts) + '\n')
        print("Done.")


if __name__ == '__main__':
    #### run trajectory editor
    # app = QApplication(sys.argv)
    # editor = TrajectoryEditor()
    # editor.show()
    # sys.exit(app.exec_())


    #### run csv2trajInfo
    # csv_path = r"/home/wicv/workspace/dig_csv_data/vehicle_info_10_20250407_10_47_34/vehicle_info_csv_10.csv"  # cut-in
    csv_path = r"/home/wicv/workspace/dig_csv_data/vehicle_cut_out_59/vehicle_cut_out_59.csv"  # cut-in&out
    agent = csv2trajInfo(csv_path, 
                         "/home/wicv/workspace/jiangchengzhi/street_gaussians/data/waymo/training/002/track/track_info.txt", 
                         "/home/wicv/workspace/jiangchengzhi/street_gaussians/data/waymo/training/002/ego_pose")
    
    # for i in agent.tracks[4]:
    #     print(i)
    agent.edit_track_info("track_info_edited2.txt")