import cv2
import numpy as np
from scipy import signal



def Thresholding_types(TH_type, image):
    processed_img = np.zeros(image.shape)

    if TH_type == 'Otsu global':
        Otsu_threshold = global_Otsu_Threshold(image)
        processed_img = Apply_global_Threshold(image, Otsu_threshold)
    else:
        if TH_type == 'Otsu local':
             processed_img = localOtsuThresholding(image, 256).astype(np.uint8)
        else:
            if TH_type == 'Optimal global':
                Optimal_G_TH = global_Optimal_Thresholding(image)
                processed_img = Apply_global_Threshold(image, Optimal_G_TH)
            else:
                if TH_type == 'Optimal local':
                    processed_img = local_Optimal_Thresholding(image, 16).astype(np.uint8)
                

    return processed_img


# helper functions
def gaussian_filter(img, shape):
    sigma = 0.5

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return signal.convolve2d(img, h)

def Apply_global_Threshold(img, threshold_value):
    row, col = img.shape
    new_img = np.zeros([row, col])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if img[i][j] >= threshold_value:
                new_img[i][j] = 255
            else:
                new_img[i][j] = 0

    new_img = new_img.astype(np.uint8)

    return new_img

def global_Otsu_Threshold(image):
    # Apply GaussianBlur to reduce image noise if it is required
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    image = gaussian_filter(image, (5, 5))
    # Set total number of bins in the histogram
    bins_num = 256
    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)
    # Get normalized histogram if it is required
    hist = np.divide(hist.ravel(), hist.max())
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    Otsu_threshold = bin_mids[:-1][index_of_max_val]

    return Otsu_threshold

def localOtsuThresholding(image,block_size):

    # resize image to square image [ max dimension] because block size is square!
    if image.shape[0] != image.shape[1]:
        if image.shape[0] > image.shape[1]:
            resizedImage = cv2.resize(image, (image.shape[0], image.shape[0]))
        else:
            resizedImage = cv2.resize(image, (image.shape[1], image.shape[1]))
    else:
        resizedImage =image
    rows = resizedImage.shape[0]
    cols = resizedImage.shape[1]

    thresholdedImage = np.zeros(resizedImage.shape)

    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            # Extarct blocks
            block = resizedImage[r:min(r + block_size, rows), c:min(c + block_size, cols)]
            blockSize = np.size(block)
            # 1 - Calculate Histogram of each block
            grayLevels = range(0, 256)
            histogram = [0] * 256
            for level in grayLevels:
                histogram[level] = len(np.extract(np.asarray(block) == grayLevels[level], block))

            # 2 - Get between class variance for each gray Level (threshold)
            betweenClassVariance = []
            for level in grayLevels:
                threshold = level
                backgroundGrayLevels = np.extract(np.asarray(grayLevels) < threshold, grayLevels)
                foregroundGrayLevels = np.extract(np.asarray(grayLevels) >= threshold, grayLevels)
                backgroundHist = []
                foregroundHist = []

                backgroundWeight = 0
                foregroundWeight = 0
                backgroundMean = 0
                foregroundMean = 0

                # get corresponding histogram for each region [ background , foreground]
                if len(backgroundGrayLevels):
                    for level in backgroundGrayLevels:
                        backgroundHist.append(histogram[level])
                        # calculate weight of background
                        backgroundWeight = float(sum(backgroundHist)) / blockSize
                    # calculate  mean of background if background exists
                    if backgroundWeight:
                        backgroundMean = np.sum(np.multiply(backgroundGrayLevels, np.asarray(backgroundHist))) / float(
                            sum(backgroundHist))

                if len(foregroundGrayLevels):
                    for level in foregroundGrayLevels:
                        foregroundHist.append(histogram[level])
                        # calculate weight of foreground
                        foregroundWeight = float(sum(foregroundHist)) / blockSize
                    # calculate  mean of foreground if foreground exists
                    if foregroundWeight:
                        foregroundMean = np.sum(np.multiply(foregroundGrayLevels, np.asarray(foregroundHist))) / float(
                            sum(foregroundHist))
                # get between class variance at current gray level
                betweenClassVariance.append(backgroundWeight * foregroundWeight * (backgroundMean - foregroundMean) * (backgroundMean - foregroundMean))

            # 3 - Get maximum gray level corresponding to maximum betweenClassVariance
            maxbetweenClassVariance = np.max(betweenClassVariance)
            otsuThreshold = betweenClassVariance.index(maxbetweenClassVariance)

            # convert to binary [ (0 , 255) only]
            thresholdedBlock = np.zeros(block.shape)
            for row in range(0, block.shape[0]):
                for col in range(0, block.shape[1]):
                    if block[row, col] >= otsuThreshold:
                        thresholdedBlock[row, col] = 255
                    else:
                        thresholdedBlock[row, col] = 0

            # fill the output image for each block
            thresholdedImage[r:min(r + block_size, rows), c:min(c + block_size, cols)] = thresholdedBlock

    # resize output  image back to original size
    thresholdedImage = cv2.resize(thresholdedImage, (image.shape[1], image.shape[0]))

    return thresholdedImage

def global_Optimal_Thresholding(image):
    rows = image.shape[0]
    cols = image.shape[1]
    # get  initial background mean  (4corners)
    background = [image[0, 0], image[0, cols-1], image[rows-1, 0], image[rows-1, cols-1]]
    background_mean = np.mean(background)
    # get  initial foreground mean
    foreground_mean = np.mean(image) - background_mean
    # get  initial threshold
    thresh = (background_mean + foreground_mean) / 2.0

    while True:
        old_thresh = thresh
        new_foreground = image[np.where(image >= thresh)]
        new_background = image[np.where(image < thresh)]
        print(new_background.size)
        if new_background.size:
            new_background_mean = np.mean(new_background)
        else:
            new_background_mean = 0
        if new_foreground.size:
            new_foreground_mean = np.mean(new_foreground)
        else:
            new_foreground_mean = 0
        # update threshold
        thresh = (new_background_mean + new_foreground_mean) / 2
        if old_thresh == thresh:
            break


    return round(thresh, 2)

def local_Optimal_Thresholding(image, block_size):
    # blockSize = 16
    if image.shape[0] != image.shape[1]:
        if image.shape[0] > image.shape[1]:
            resizedImage = cv2.resize(image, (image.shape[0], image.shape[0]))
        else:
            resizedImage = cv2.resize(image, (image.shape[1], image.shape[1]))
    else:
        resizedImage = image
    rows = resizedImage.shape[0]
    cols = resizedImage.shape[1]
    outputImage = np.zeros(resizedImage.shape)

    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            # Extarct blocks
            block = resizedImage[r:min(r + block_size,rows), c:min(c + block_size, cols)]
            # get  initial background mean  (4corners)
            background = [block[0, 0], block[0, block.shape[1]-1], block[block.shape[0]-1, 0], block[block.shape[0]-1, block.shape[1]-1]]
            background_mean = np.mean(background)
            # get  initial foreground mean
            foreground_mean = np.mean(block) - background_mean
            # get  initial threshold
            thresh = (background_mean + foreground_mean) / 2.0

            while True:
                old_thresh = thresh
                new_foreground = block[np.where(block >= thresh)]
                new_background = block[np.where(block < thresh)]
                if new_background.size:
                    new_background_mean = np.mean(new_background)
                else:
                    new_background_mean = 0
                if new_foreground.size:
                    new_foreground_mean = np.mean(new_foreground)
                else:
                    new_foreground_mean = 0
                # update threshold
                thresh = (new_background_mean + new_foreground_mean) / 2
                if old_thresh == thresh:
                    break

            # convert to binary [ (0 , 255) only]
            thresholdedBlock = np.zeros(block.shape)
            for row in range(0, block.shape[0]):
                for col in range(0, block.shape[1]):
                    if block[row, col] >= thresh:
                        thresholdedBlock[row, col] = 255
                    else:
                        thresholdedBlock[row, col] = 0

            # fill the output image for each block
            outputImage[r:min(r + block_size, rows), c:min(c + block_size, cols)] = thresholdedBlock

    # resize output  image back to original size
    outputImage = cv2.resize(outputImage, (image.shape[1], image.shape[0]))

    return outputImage


### ................test...................... ###

# # Read the image in a grayscale mode
# image = cv2.imread('test_image/Otsu_boat.jpg', 0)
#
# test = Thresholding_types('Optimal_L', image)
#
# cv2.imshow("test", test)
# cv2.waitKey(0)