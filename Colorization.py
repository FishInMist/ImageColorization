# Author: xwang875

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import linalg


def colorization(originalImage, markedImage, fileName, cutoffMode):
    # obtain the coordinates of the scribbles in the image
    markPos = np.sum(np.absolute(originalImage - markedImage), axis=2) > 0
    cv2.imwrite(fileName + '_markLocation.bmp', markPos * 255)

    # convert the BW and Mark BW image into YUV channel
    convertedImageO = cv2.cvtColor(originalImage, cv2.COLOR_BGR2YUV)
    convertedImageM = cv2.cvtColor(markedImage, cv2.COLOR_BGR2YUV)

    # auxiliary image have the same intensity with the original image but include the mark color from the marked image
    Y = convertedImageO[:, :, 0]  # intensity channel Y from original image
    U = convertedImageM[:, :, 1]  # channel U
    V = convertedImageM[:, :, 2]  # channel V

    # W matrix calculation
    windowWidth = 1  # set the width of window about neighbours s around site r
    numOfPixels = (2 * windowWidth + 1) ** 2

    row = Y.shape[0]
    col = Y.shape[1]
    totalPos = row * col

    # Wrs should be weight between point r and s, in order to simply the process,
    # only neighbors in the window is considered
    # i.e. each point r should have (numOfPixels - 1) s, plus weight of r itself
    rIndexes = np.zeros(totalPos * numOfPixels)
    sIndexes = np.zeros(totalPos * numOfPixels)
    wrs = np.zeros(totalPos * numOfPixels, dtype=np.double)  # matrix to save Wrs
    kernel = np.zeros(numOfPixels, dtype=np.double)  # kernel matrix for each window around pixel r
    imageIndex = np.arange(totalPos).reshape((row, col))
    # process the pixels in image one by one, with the given window size
    windowRange = range(-windowWidth, windowWidth + 1)

    counter = 0
    for r in range(row):
        for c in range(col):
            # if the pixel is not marked by color
            index = r * col + c
            if not markPos[r, c]:
                kernelCounter = 0
                for x in windowRange:
                    for y in windowRange:
                        xPos = r + x
                        yPos = c + y
                        if xPos < 0 or xPos >= row or yPos < 0 or yPos >= col:  # out of image bounds
                            continue
                        if x == 0 and y == 0:  # r is not neighbor of r pixel
                            continue
                        kernel[kernelCounter] = Y[xPos, yPos]
                        rIndexes[counter] = index
                        sIndexes[counter] = imageIndex[xPos, yPos]
                        kernelCounter += 1
                        counter += 1
                yr = Y[r, c]
                kernel[kernelCounter] = yr
                var = np.var(kernel[0:kernelCounter + 1])

                # due to the data type overflow, need to set a cutoff for the variance value
                # Please see the implementation below by myself, it didn't show very good result
                if cutoffMode == 0:
                    variance = 2 * var
                    cutoff = 0.1
                    if variance < cutoff:
                        variance = cutoff

                # Also, I referenced the source MATLAB code for the variance cutoff as comparison with my implementation
                # the comparison is discussed in the report
                elif cutoffMode == 1:
                    variance = 0.6 * var
                    minValue = ((kernel[0:kernelCounter] - yr) ** 2).min()
                    if variance < minValue / 2.0:
                        variance = minValue / 2.0
                    cutoff = 0.000002
                    if variance < cutoff:
                        variance = cutoff

                # equation (2) in source paper
                kernel[0:kernelCounter] = np.exp(-((kernel[0:kernelCounter] - yr) ** 2) / variance)
                # normalize the wrs so that they sum to 1
                kernel[0:kernelCounter] = kernel[0:kernelCounter] / kernel[0:kernelCounter].sum()
                # D - W = D + (-W), therefore, this step, I fill -wrs into the matrix
                wrs[counter - kernelCounter:counter] = -kernel[0:kernelCounter]
            # fill the diagonal element with 1
            rIndexes[counter] = index
            sIndexes[counter] = imageIndex[r, c]
            wrs[counter] = 1
            counter += 1

    # Wrs stands for (D - W) here.
    # Considering the scribble pixels that didn't consider neighbor pixels,
    # the total number of values in wrs might be less than totalPos * numOfPixels,
    # therefore, to keep only valid values, the not filled portion should be discarded.
    wrs = wrs[0: counter]
    rIndexes = rIndexes[0:counter]
    sIndexes = sIndexes[0:counter]

    # Generate the sparse matrix
    Wrs = coo_matrix((wrs, (rIndexes, sIndexes)), shape=(totalPos, totalPos), dtype=np.float64)
    # solve the function (D - W) x = b
    newU = linalg.spsolve(Wrs, (U * markPos).flatten())
    newV = linalg.spsolve(Wrs, (V * markPos).flatten())

    output = np.zeros((row, col, 3), np.uint8)
    output[:, :, 0] = Y
    output[:, :, 1] = newU.reshape((row, col))
    output[:, :, 2] = newV.reshape((row, col))
    cv2.imwrite(fileName + '_res.bmp', cv2.cvtColor(output, cv2.COLOR_YUV2BGR))


