from PIL import Image
from scipy.misc import imsave
import numpy
import glob, os
from multiprocessing import Pool
import cv2


folder_names = ['densepose', 'PNCC', '3dTex', 'face']
base_name = '/data1/cuiqingli/data/render_v5/unzipV3'
save_base_name = '/data1/cuiqingli/data/render_v5/denseposeV4_256'
def handler(folder_id):
	dirname = os.path.join(base_name, str(folder_id))
	save_dirname = os.path.join(save_base_name, str(folder_id))

	for folder_name in folder_names:
		folder_path = os.path.join(dirname, folder_name)
		save_folder_path = os.path.join(save_dirname, folder_name)

		for file in os.listdir(folder_path):
			if file.endswith('.png'):
				array = file.split(".")
				image_id = array[0].split("_")[1]
				image_id = int(image_id)

				read_image = cv2.imread(os.path.join(folder_path, file))
				print(read_image.shape)
				down_px = 0
				read_image = read_image[400-down_px :1120- down_px , 0:720]

				read_image = cv2.resize(read_image, (256, 256))
				save_image_path = os.path.join(save_folder_path, file)
				print(save_image_path)
				cv2.imwrite(save_image_path, read_image)









# handler(1)

lines = [i+1 for i in range(100)]


pool = Pool(42)
pool.map(handler, lines)

