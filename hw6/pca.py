import numpy as np
import skimage
import os, sys

from os.path import basename
from skimage import io


def recon(img):
	M = img- np.min(img)
	M /= np.max(M)
	M = (M*255).astype(np.uint8)
	M = np.clip(M,0,255)

	return M
def read_data(folder):
	print("Loading data")
	image_list = sorted(os.listdir(folder),key=lambda x: int(basename(x).split('.')[0]))
	data = np.zeros((len(image_list), 1080000))

	for i in range(len(image_list)):
		image = io.imread(os.path.join(folder,image_list[i]))
		data[i] = np.reshape(image, (1, 1080000))

	return data

def face_recon(img_len,weights,mean,U,folder, idx):
	face = mean + np.dot(weights, U[:,:4].T)
	img = []
	for i in range(img_len):
		img.append(recon(face[i]))

	print("reconstructing")
	face_idx = int(basename(os.path.join(folder,idx)).split(".")[0])

	rec_image = np.reshape(img[face_idx], (600, 600, 3)).astype(np.uint8)
	io.imsave("reconstruction.jpg", rec_image)

def main(argv):

	data = read_data(argv[1])
	mean = np.mean(data, axis = 0)

	print("SVD .....")
	U, s, V = np.linalg.svd((data - mean).T, full_matrices = False)

	# eg face index
	weights = np.dot((data - mean), U)[:,:4]
	face_recon(data.shape[0],weights,mean,U,argv[1],argv[2])



if __name__=='__main__':
    main(sys.argv)
