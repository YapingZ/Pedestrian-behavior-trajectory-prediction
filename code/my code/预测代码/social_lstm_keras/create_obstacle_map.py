

import numpy as np
import pickle
import matplotlib.pyplot as plt


def convert_to_obstacle_map(img):
    '''
    Function to create an obstacle map from the annotaetd image
    params:
    img : Image file path
    '''
    im = plt.imread(img)
    # im is a numpy array of shape (w, h, 4)
    w = im.shape[0]
    h = im.shape[1]

    obs_map = np.ones((w, h))

    for i in range(w):
        for j in range(h):
            # rgba is a 4-dimensional vector
            rgba = im[i, j]
            # obstacle
            if rgba[0] == 0 and rgba[1] == 0 and rgba[2] == 0:
                # print "Obstacle found"
                obs_map[i, j] = 0
            # Partially traversable
            elif rgba[0] == 0 and rgba[1] == 0:
                # print "Partially traversable found"
                obs_map[i, j] = 0.5

    return obs_map


def main():
    data_dirs = ['../data/eth/univ', '../data/eth/hotel',
                 '../data/ucy/zara/zara01', '../data/ucy/zara/zara02',
                 '../data/ucy/univ']

    for x in data_dirs:
        image_file = x + '/annotated.png'
        obs_map = convert_to_obstacle_map(image_file)
        f = open(x+'/obs_map.pkl', 'wb')
        pickle.dump(obs_map, f, protocol=2)
        f.close()

if __name__ == '__main__':
    main()