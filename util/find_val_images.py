import numpy as np
import random
import glob
import os
from utils import verify_if_folder_exist_or_create_it


def count_lines_in_file(file_path):
    try:
        images = []
        with open(file_path, 'r') as file:
            line_count = 0
            for line in file:
                image = line.split(" ")[0].split("/")[-1]
                images.append(image)
        return images
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return -1  # Return -1 or handle the error according to your

dataset = "potsdam"
size_patches = 513
split_name = f"1_2_{size_patches}"
path_images = f"2_Ortho_RGB_{size_patches}/"
path_gts  = f"5_Labels_all_{size_patches}/"

list_images = glob.glob(f"./{dataset}/{path_images}*")
total_number_images = len(list_images)
print(total_number_images)

path = f"./splits/{dataset}/1_2_513/labeled.txt"
labeled_images = count_lines_in_file(path)

path = f"./splits/{dataset}/1_2_513/unlabeled.txt"
unlabeled_images = count_lines_in_file(path)

count = 0
with open(f"./splits/{dataset}/{split_name}/val_test.txt", "w") as file:
    for image in list_images:
        image_name = image.split("/")[-1]
        if image_name in labeled_images:
            continue
        elif image_name in unlabeled_images:
            continue
        else:
            count += 1
            gt_image_name = image_name.replace("_RGB.png","_label.png")
            file.write(f"2_Ortho_RGB_513/{image_name} 5_Labels_all_513/{gt_image_name}\n")
print(count)


# with open(f"./splits/{dataset}/{split_name}/labeled.txt", "w") as file:
#     for index, image in enumerate(list_images):
#         if index <= int(total_number_images*train_percentage)*label_percentage:
#             img_file_name = image.split("/")[-1].split("_")[:-1]
#             file.write(f"{path_images}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_{img_file_name[4]}_{img_file_name[5]}_RGB.png {path_gts}{img_file_name[0]}_{img_file_name[1]}_{img_file_name[2]}_{img_file_name[3]}_{img_file_name[4]}_{img_file_name[5]}_label.png\n")