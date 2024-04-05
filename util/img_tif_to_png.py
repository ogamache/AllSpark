import numpy as np
from PIL import Image
import glob
from tqdm import tqdm


# path_imgs = "./VOC2012/SegmentationClass/"

# unique_value_labels = [29, 76, 150, 179, 226]
# for idx, img in enumerate(tqdm(glob.glob(f"{path_imgs}*"))):
#     if idx == 0:
#         image = np.array(Image.open(img))
#         print(img)
#         print(np.unique(image))

# GT images (Potsdam)
path_imgs = "./potsdam/5_Labels_all_patches/"

unique_value_labels = [29, 76, 150, 179, 226, 255]
for idx, img in enumerate(tqdm(glob.glob(f"{path_imgs}*"))):
    if img.endswith(".tif"):
        print("###########")
        print(img)
        image = np.array(Image.open(img))
        print(np.unique(image))
        for idx, value in enumerate(unique_value_labels):
            # image[np.logical_and(image >= value-10,image <= value+10)] = idx
            image[image == value] = idx
        print(np.unique(image))
        # image = Image.fromarray(image)
        # new_file_name = img.replace(".tif", ".png")
        # image.save(new_file_name)

# # images
# path_imgs = "./potsdam/2_Ortho_RGB_patches/"

# for img in tqdm(glob.glob(f"{path_imgs}*")):
#     if img.endswith(".tif"):
#         image = Image.open(img)
#         new_file_name = img.replace(".tif", ".png")
#         image.save(new_file_name)