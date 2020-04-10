import os

path_images = r'D:\dev\ms_data\Challenges\ISBI2015\ISBI_L\05'


folders = os.listdir(path_images)
i = 1
for folder in folders:
    files = os.listdir(os.path.join(path_images, folder))
    for file in files:
        format = file[-7:]
        image_name = file.split(".")[0]
        os.rename(os.path.join(path_images,folder, file), os.path.join(path_images, folder, image_name + "_"+ str(i).zfill(2) + format))
    i += 1