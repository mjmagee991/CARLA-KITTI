import sys
import os
import numpy as np
import csv
from scipy.spatial.transform import Rotation

data_path = "object/training"

def get_labels(filepath):
    with open(filepath) as label_file:
        labels = []
        reader = csv.reader(label_file, delimiter=' ', skipinitialspace=True)
        for line in reader:
            # Ignore values that aren't changed by the reference frame
            ignore_str = ""
            for i in range(11):
                ignore_str += ' ' + line[i]
            ignore_str = ignore_str.lstrip()

            # Get the position and rotation
            position = [float(line[11]), float(line[12]), float(line[13])]
            yaw_rotation = float(line[14])
            labels.append([ignore_str, position, yaw_rotation])
            
    return labels


def main():
    # Validate arguments
    if len(sys.argv) != 2:
        print("Incorrect arguments")
        exit()

    # For all data##/ directories
    data_root = sys.argv[1]
    data_dirs = [d for d in os.scandir(data_root)]
    for d in os.scandir(data_root):
        if not d.is_dir():
            print("File found in top-level directory, where only subdirectories are expected")
            print(f"'{d.name}' is not a directory")
            exit()

        # Define all input and output directories
        pose_dir = os.path.join(d, f"{data_path}/pose")
        ego_point_cloud_dir = os.path.join(d, f"{data_path}/velodyne")
        rsu_point_cloud_dir = os.path.join(d, f"{data_path}/velodyne_rsu")
        fused_point_cloud_dir = os.path.join(d, f"{data_path}/velodyne_fused")
        ego_label_dir = os.path.join(d, f"{data_path}/label_2")
        rsu_label_dir = os.path.join(d, f"{data_path}/rsu_label_2")
        fused_label_dir = os.path.join(d, f"{data_path}/fused_label_2")

        # Create output directories
        os.makedirs(fused_point_cloud_dir)
        os.makedirs(fused_label_dir)

        # Create RSU-world transformation matrix
        # No rotation because it is already handled in data_collector.py
        world_from_rsu = np.eye(4)
        world_from_rsu[:3, 3] = [-93.5, 145.5, 5.5]

        # Combine point clouds and labels for each file in the directory
        for filename in [f.name[:-4] for f in os.scandir(ego_point_cloud_dir)]:
            # Read pose data and create transformation
            with open(f"{pose_dir}/{filename}.txt") as pose_file:
                reader = csv.reader(pose_file, skipinitialspace=True)
                pose_data = [float(s) for s in next(reader)]

            # ego_body is the reference of the body of the ego vehicle
            world_from_ego_body = np.eye(4)
            world_from_ego_body[:3, :3] = Rotation.from_euler('ZYX', [pose_data[3], -pose_data[4], -pose_data[5]], degrees=True).as_matrix()
            world_from_ego_body[:3, 3] = pose_data[:3]
            # ego is the reference frame of the ego LiDAR
            ego_body_from_ego = np.eye(4)
            ego_body_from_ego[:3, 3] = [1.6, 0, 1.7]

            world_from_ego = world_from_ego_body @ ego_body_from_ego
            ego_yaw_rotation = np.deg2rad(pose_data[3])

            # Calculate transformation from rsu to ego coordinates
            ego_from_world = np.linalg.inv(world_from_ego)
            ego_from_rsu = ego_from_world @ world_from_rsu
            # Transpose here because the point clouds store the points as row vectors, not column vectors
            rsu_rotate_to_ego = np.transpose(ego_from_rsu[:3, :3])
            rsu_translate_to_ego = ego_from_rsu[:3, 3]

            # Read combine point clouds
            ego_point_cloud = np.fromfile(f"{ego_point_cloud_dir}/{filename}.bin", dtype=np.float32).reshape(-1, 4)
            rsu_point_cloud = np.fromfile(f"{rsu_point_cloud_dir}/{filename}.bin", dtype=np.float32).reshape(-1, 4)
            
            # RSU point cloud needs to have the y-axis inverted because CARLA and Unreal use different axes
            # See docs for more detail
            rsu_point_cloud[:, 1] = -rsu_point_cloud[:, 1]
            rsu_point_cloud[:, :3] @= rsu_rotate_to_ego
            rsu_point_cloud[:, :3] += rsu_translate_to_ego
            rsu_point_cloud[:, 1] = -rsu_point_cloud[:, 1]

            fused_point_cloud = np.concatenate((ego_point_cloud, rsu_point_cloud), axis=0)
            fused_point_cloud.tofile(f"{fused_point_cloud_dir}/{filename}.bin")

            # Read and combine labels
            ego_labels = get_labels(f"{ego_label_dir}/{filename}.txt")
            rsu_labels = get_labels(f"{rsu_label_dir}/{filename}.txt")

            # Transform RSU labels to the ego coordinate frame
            for i in range(len(rsu_labels)):
                # Label data is stored as (y, -z, x) but our transform has to be done in (x, y, z)
                # See docs for more detail
                label = rsu_labels[i][1]
                position = [label[2], label[0], -label[1]]
                position @= rsu_rotate_to_ego
                position += rsu_translate_to_ego
                rsu_labels[i][1] = [position[1], -position[2], position[0]]

                rsu_labels[i][2] -= ego_yaw_rotation

            # Remove duplicates and the ego vehicle bounding box
            for ego_label in ego_labels:
                for rsu_label in rsu_labels:
                    if np.linalg.norm(np.subtract([0, 1.7, -1.6], rsu_label[1])) < 0.1 or np.linalg.norm(np.subtract(ego_label[1], rsu_label[1])) < 0.1:
                        rsu_labels.remove(rsu_label)
                        break

            # Save fused labels to a file
            fused_labels = ego_labels + rsu_labels
            with open(f"{fused_label_dir}/{filename}.txt", "w") as fused_label_file:
                for label in fused_labels:
                    fused_label_file.write(f"{label[0]} {label[1][0]} {label[1][1]} {label[1][2]} {label[2]}\n")

if __name__ == '__main__':
    main()
