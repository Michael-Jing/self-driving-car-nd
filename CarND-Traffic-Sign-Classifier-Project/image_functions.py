import numpy as np
import cv2

# this function is from  https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.jqnd5bnzd
def augment_brightness_camera_images(image):
    image = image.astype(np.uint8)
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    # print(random_bright)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

# this function is from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.jqnd5bnzd
def add_random_shadow(image):
    image = image.astype(np.uint8)
    top_y = 32*np.random.uniform()
    top_x = 0
    bot_x = 16
    bot_y = 32*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def crop(img, num_pixel, position):

    new_img = np.zeros(img.shape).astype(np.uint8)

    if position == "top":
        new_img[0:num_pixel] = np.copy(img[0])
        new_img[num_pixel:img.shape[0]] = np.copy(img[0:img.shape[0] - num_pixel])
    elif position == "bottom":
        new_img[0:img.shape[0] - num_pixel] = img[num_pixel:img.shape[0]]
        new_img[img.shape[0] - num_pixel: img.shape[0]] = img[img.shape[0] - 1]
    elif position == "left":
        boader = img[:, 0, :]
        boader = boader.reshape((boader.shape[0], 1, boader.shape[1]))
        new_img[:,0:num_pixel,:] = boader
        new_img[:,num_pixel:img.shape[1],:] = img[:,0:img.shape[1] - num_pixel,:]

    elif position == "right":
        boader = img[:, img.shape[1] - 1, :]
        boader = boader.reshape((boader.shape[0], 1, boader.shape[1]))
        new_img[:, 0:img.shape[1] - num_pixel, :] = img[:, num_pixel: img.shape[1], :]
        new_img[:, img.shape[1] - num_pixel : img.shape[1], :] = boader

    elif position == "top_left":
        temp = crop(img, num_pixel, "top")
        new_img = crop(temp, num_pixel, "left")

    elif position == "top_right":
        temp = crop(img, num_pixel, "top")
        new_img = crop(temp, num_pixel, "right")
    elif position == "bottom_left":
        temp = crop(img, num_pixel, "bottom")
        new_img = crop(temp, num_pixel, "left")

    elif position == "bottom_right":
        temp = crop(img, num_pixel, "bottom")
        new_img = crop(temp, num_pixel, "right")
    return new_img

