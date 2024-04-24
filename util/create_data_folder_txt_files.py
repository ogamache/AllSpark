import numpy as np
import random
import glob
import os
from utils import verify_if_folder_exist_or_create_it

dataset = "potsdam"
split_name = "1_2_513"
path_images = "2_Ortho_RGB_513/"
path_gts  = "5_Labels_all_513/"
train_percentage = 0.8
label_percentage = 0.5

verify_if_folder_exist_or_create_it(f"./splits/{dataset}/{split_name}/")
list_images = glob.glob(f"./{dataset}/{path_gts}*")
total_number_images = len(list_images)
random.shuffle(list_images)

with open(f"./splits/{dataset}/{split_name}/labeled.txt", "w") as file:
    for index, image in enumerate(list_images):
        if index <= int(total_number_images*train_percentage)*label_percentage:
            img_file_name = image.split("/")[-1].split("_")[:-1]
            file.write(f"{path_images}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_{img_file_name[4]}_{img_file_name[5]}_RGB.png {path_gts}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_{img_file_name[4]}_{img_file_name[5]}_label.png\n")

with open(f"./splits/{dataset}/{split_name}/unlabeled.txt", "w") as file:
    for index, image in enumerate(list_images):
        if index > int(total_number_images*train_percentage)*label_percentage and index <= int(total_number_images*train_percentage):
            img_file_name = image.split("/")[-1].split("_")[:-1]
            file.write(f"{path_images}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_{img_file_name[4]}_{img_file_name[5]}_RGB.png {path_gts}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_{img_file_name[4]}_{img_file_name[5]}_label.png\n")

with open(f"./splits/{dataset}/val.txt", "w") as file:
    for index, image in enumerate(list_images):
        if index > int(total_number_images*train_percentage):
            img_file_name = image.split("/")[-1].split("_")[:-1]
            file.write(f"{path_images}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_{img_file_name[4]}_{img_file_name[5]}_RGB.png {path_gts}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_{img_file_name[4]}_{img_file_name[5]}_label.png\n")