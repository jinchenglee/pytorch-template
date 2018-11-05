import numpy as np
import time
from scipy.ndimage import binary_dilation

#from  multiprocessing.dummy import Pool as ThreadPool # THIS IS NOT RIGHT, single process multi-thread, not friendly to Python
from multiprocessing import Pool as ProcessPool # This is real multi-process/multi-thread.

def dilate_k_pix_threads(masks, threads=2):
    pool = ProcessPool(threads)
    #results = pool.map(dilate_k_pix, masks)
    results = pool.map(dilate_k_pix_ndimage, masks)
    pool.close()
    pool.join()
    return results

def manhattan_dist(mask):
    """
    Algorithm to return manhattan distance to nearest active pixel in input mask
    in O(n^2) time. 
    
    Algorithm description: https://blog.ostermiller.org/dilate-and-erode

    Args:
        mask: binary mask image (H x W) where 1 shows active pixel, 0 otherwise.

    Return:
        man_dist: numpy array of same size (H x W) as mask, number representing
            Manhattan distance to nearest active pixel.
    """
    # time_start = time.time()

    h, w = mask.shape

    # Default distance: at most image height + width
    man_dist = np.ones_like(mask) * (h + w - 2)

    # Traverse from top-left to bottom-right
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1:
                man_dist[i, j] = 0
            else:
                # Compare to north/west neighbors 
                if (i>0):
                    man_dist[i, j] = min(man_dist[i, j], man_dist[i-1, j]+1)
                if (j>0):
                    man_dist[i, j] = min(man_dist[i, j], man_dist[i, j-1]+1)    
    
    # Traverse in reverse direction
    for i in reversed(range(h)):
        for j in reversed(range(w)):
            # Compare to south/east neighbors 
            if (i<h-1):
                man_dist[i, j] = min(man_dist[i, j], man_dist[i+1, j]+1)
            if (j<w-1):
                man_dist[i, j] = min(man_dist[i, j], man_dist[i, j+1]+1)

    # print("manhattan_dist() time used:", time.time() - time_start)
    return man_dist

def dilate_k_pix(mask, k=4):
    """
    Dilate by k pixels around active pixel of input mask.

    Args:
        mask: binary mask image (H x W) where 1 shows active pixel, 0 otherwise.

    Return:
        mask_dilated: numpy array of same size (H x W) as binary mask, dilated.
    """
    # time_start = time.time()

    mask_man_dist = manhattan_dist(mask)

    mask_dilated = mask.copy()

    h, w = mask.shape

    for i in range(h):
        for j in range(w):
            if mask_man_dist[i, j] <= k:
                mask_dilated[i, j] = 1

    # print("dilate_k_pix() time used:", time.time() - time_start)
    return mask_dilated
    
def dilate_k_pix_ndimage(mask, k=4):
    """
    Dilate by k pixels around active pixel of input mask. Scipy.ndimage implementation.
      30x faster than the manual implementation in Python above. 

    Args:
        mask: binary mask image (H x W) where 1 shows active pixel, 0 otherwise.

    Return:
        mask_dilated: numpy array of same size (H x W) as binary mask, dilated.
    """
    # time_start = time.time()

    mask_dilated = mask.copy()
    for i in range(k):
        mask_dilated = binary_dilation(mask_dilated)

    # print("dilate_k_pix_ndimage() time used:", time.time() - time_start)
    return mask_dilated
