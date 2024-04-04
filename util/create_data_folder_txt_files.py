import numpy as np
import random
import glob
import os

dataset = "potsdam"
split_name = "1_2"
path_images = "2_Ortho_RGB/"
path_gts  = "5_Labels_all/"
train_percentage = 0.7

def verify_if_folder_exist_or_create_it(path):
        if not os.path.isdir(path):
            os.makedirs(path)

verify_if_folder_exist_or_create_it(f"./splits/{dataset}/{split_name}/")
list_images = glob.glob(f"./{dataset}/{path_gts}*")
total_number_images = len(list_images)
random.shuffle(list_images)

with open(f"./splits/{dataset}/{split_name}/labeled.txt", "w") as file:
    for index, image in enumerate(list_images):
        if index <= int(total_number_images*train_percentage)/2:
            img_file_name = image.split("/")[-1].split("_")[:-1]
            file.write(f"{path_images}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_RGB.png {path_gts}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_label.png\n")

with open(f"./splits/{dataset}/{split_name}/unlabeled.txt", "w") as file:
    for index, image in enumerate(list_images):
        if index > int(total_number_images*train_percentage)/2 and index <= int(total_number_images*train_percentage):
            img_file_name = image.split("/")[-1].split("_")[:-1]
            file.write(f"{path_images}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_RGB.png {path_gts}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_label.png\n")

with open(f"./splits/{dataset}/val.txt", "w") as file:
    for index, image in enumerate(list_images):
        if index > int(total_number_images*train_percentage):
            img_file_name = image.split("/")[-1].split("_")[:-1]
            file.write(f"{path_images}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_RGB.png {path_gts}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_label.png\n")