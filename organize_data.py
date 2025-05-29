import sys
import os
import random
import shutil

new_folder_prefix = "my_kitti"
data_path = "object/training"
testing_dirs = [("calib", "txt"), ("image_2", "png"), ("velodyne", "bin")]
training_dirs = [("calib", "txt"), ("image_2", "png"), ("label_2", "txt"), ("velodyne", "bin")]

def get_file_list(d):
    d = d.path
    path = os.path.join(d, f"{data_path}/velodyne")
    paths = [f for f in os.scandir(path)]

    file_tuples = [(d, f.name[:-4]) for f in paths]

    return file_tuples

def main():
    # Validate arguments
    if len(sys.argv) != 2:
        print("Incorrect arguments")
        exit()

    data_root = sys.argv[1]
    data_dirs = [d for d in os.scandir(data_root)]
    if any(not p.is_dir() for p in data_dirs):
        print("File in top-level directory")
        exit()

    # Create a list of all files in all data directories
    files = []
    for d in data_dirs:
        files += get_file_list(d)

    # Randomly shuffle the files
    random.shuffle(files)

    # Split into a test set and training set
    split_point = len(files) // 2
    training_files = files[:split_point]
    testing_files = files[split_point:]

    # Create all output directories
    os.makedirs(f"{new_folder_prefix}/ImageSets")
    for td in testing_dirs:
        os.makedirs(f"{new_folder_prefix}/testing/{td[0]}")
    for td in training_dirs:
        os.makedirs(f"{new_folder_prefix}/training/{td[0]}")

    # Copy data files to output directories, according to their assigned set
    for i, tf in enumerate(testing_files):
        for td in testing_dirs:
            shutil.copy(f"{tf[0]}/{data_path}/{td[0]}/{tf[1]}.{td[1]}", f"{new_folder_prefix}/testing/{td[0]}/{i:06d}.{td[1]}")
    for i, tf in enumerate(training_files):
        for td in training_dirs:
            shutil.copy(f"{tf[0]}/{data_path}/{td[0]}/{tf[1]}.{td[1]}", f"{new_folder_prefix}/training/{td[0]}/{i:06d}.{td[1]}")


    # Generate image set text files to identify data
    with open(f"{new_folder_prefix}/ImageSets/test.txt", "w") as f:
        for i in range(len(testing_files)):
            f.write(f"{i:06d}\n")

    train_indices = random.sample(range(len(training_files)), len(training_files) // 2)
    with open(f"{new_folder_prefix}/ImageSets/trainval.txt", "w") as tv, open(f"{new_folder_prefix}/ImageSets/train.txt", "w") as t, open(f"{new_folder_prefix}/ImageSets/val.txt", "w") as v:
        for i in range(len(training_files)):
            tv.write(f"{i:06d}\n")
            if i in train_indices:
                t.write(f"{i:06d}\n")
            else:
                v.write(f"{i:06d}\n")

if __name__ == '__main__':
    main()
