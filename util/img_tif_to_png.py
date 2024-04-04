import numpy as np
from PIL import Image
import glob
from tqdm import tqdm


path_imgs = "./potsdam/5_Labels_all/"

unique_value_labels = [29, 76, 150, 179, 226]
for img in tqdm(glob.glob(f"{path_imgs}*")):
    if img.endswith(".tif"):
        image = np.array(Image.open(img).convert("L"))
        for idx, value in enumerate(unique_value_labels):
            image[image == value] = idx
        # print(np.unique(image))
        image = Image.fromarray(image)
        new_file_name = img.replace(".tif", ".png")
        image.save(new_file_name)