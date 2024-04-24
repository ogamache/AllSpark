import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from utils import verify_if_folder_exist_or_create_it

#Transforms tif images to numpy array, the takes sections of 513x513 for each color and nir, saves them in a file

#Create the paths to the folders, list the directory in the large image folder
image_folders_path = "./potsdam/2_Ortho_RGB/"
gt_folders_path = "./potsdam/5_Labels_all/"
size_patches = 513

output_images_folder_path = "./potsdam/2_Ortho_RGB_513/"
output_gt_folder_path = "./potsdam/5_Labels_all_513/"
verify_if_folder_exist_or_create_it(output_images_folder_path)
verify_if_folder_exist_or_create_it(output_gt_folder_path)

image_folders = os.listdir(image_folders_path)
unique_value_labels = [29, 76, 150, 179, 226, 255]

#Removes maximum size for image reading
Image.MAX_IMAGE_PIXELS = None

#Iterate over each folder containing an image, transforms tif into numpy array

for image in tqdm(image_folders):
    if image.endswith(".tif"):
        gt_image = image.replace("_RGB", "_label")

        img = np.array(Image.open(f"{image_folders_path}/{image}"))
        gt_img = np.array(Image.open(f"{gt_folders_path}/{gt_image}").convert('L'))
        for idx, value in enumerate(unique_value_labels):
            gt_img[np.logical_and(gt_img >= value-2,gt_img <= value+2)] = idx
            # gt_img[gt_img == value] = idx
        # print(gt_image)
        # print(np.unique(gt_img))

        #Number of horizontal and vertical patches that fit in large image
        n_patches_hor = np.shape(img)[0]//size_patches
        n_patches_vert = np.shape(img)[1]//size_patches

        #Iterate over number of patches, takes patches of 512x512
        for i in range(n_patches_hor):
            for j in range(n_patches_vert):

                patch_img = Image.fromarray(img[0+size_patches*i:size_patches+size_patches*i, 0+size_patches*j:size_patches+size_patches*j])
                patch_gt_img = Image.fromarray(gt_img[0+size_patches*i:size_patches+size_patches*i, 0+size_patches*j:size_patches+size_patches*j])

                #Save the patches in the same folder
                new_img_name = image.replace("RGB.tif", f"{i}_{j}_RGB.png")
                new_gt_img_name = gt_image.replace("label.tif", f"{i}_{j}_label.png")
                patch_img.save(f"{output_images_folder_path}/{new_img_name}")
                patch_gt_img.save(f"{output_gt_folder_path}/{new_gt_img_name}")

