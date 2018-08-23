# K-Nearest Neighbor Algorithm

import math
import operator
import struct
import numpy as np
import cv2
import skimage.measure as measure
from scipy.ndimage.interpolation import geometric_transform

#------------------  Configuration ------------------ #

# Number of images to be used from Training Data Set
TRAINING_LIMIT = 100
# Number of images to be used from Test Data Set
TEST_LIMIT = 5

# K - Number of nearest neighbors
k = 3

# Rotation angle for test images
rotationAngle = 90

#------------------ Methods ------------------ #

# Read IDX File
def readIDX(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

# Read training and test data sets from input files
def getInputFiles():
    # Read Training DataSets
    raw_train_data = readIDX("input/train-images-idx3-ubyte")
    trainingSet = np.reshape(raw_train_data, (60000, 28 * 28))
    train_label = readIDX("input/train-labels-idx1-ubyte")

    # Read Testing DataSets
    raw_test_data = readIDX("input/t10k-images-idx3-ubyte")
    testSet = np.reshape(raw_test_data, (10000, 28 * 28))
    test_label = readIDX("input/t10k-labels-idx1-ubyte")

    return trainingSet, train_label, testSet, test_label

# Rotate image
def rotateImage90Degrees(image, degree):
    if(degree != 0):
        # Reshape the pixels to 28 X 28 Matrix
        rotatedImage = np.reshape(image, [28, 28])

        # Rotate the image
        for i in range(int(degree / 90)):
            rotatedImage = list(zip(*reversed(rotatedImage)))

        # Convert the 28 X 28 image to 1 X 784
        image = np.reshape(rotatedImage, 28 * 28)
    return image

# Calculate accuracy of predictions
def getAccuracy(testLabel, predictions):
    correct = 0
    for x in range(len(testLabel)):
        #  Comparison between test label data and predicted label data through knn
        if testLabel[x] == predictions[x]:
            correct += 1

    # Computing percentage of correctly predicted labels with total test labels
    return (correct / float(len(testLabel))) * 100.0

# Calculate Euclidean Distance
def euclideanDistance(A, B):
    # Difference between pixel values
    diff = [a - b for a, b in zip(A, B)]
    # Square of the calculated difference and summation of those squares
    distance = sum([x ** 2 for x in diff])
    # Square root of the summation
    return math.sqrt(distance)

# Calculate Mahalanobis Distance
def mahalanobisDistance(x, y):
    #  Find Covariance
    covariance_xy = np.cov(x, y, rowvar=0)
    # Find inverse covariance
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    # Find mean
    xy_mean = np.mean(x), np.mean(y)
    # Difference between pixel and the mean for each value
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    # transpose of the difference calculated
    diff_xy = np.transpose([x_diff, y_diff])

    mahalanobisDistance = 0
    for i in range(len(diff_xy)):
        # Summation of the Square root of the dot product of the dot product of transposed matrix with inverse covariance matrix with transposed matrix
        mahalanobisDistance += np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]), inv_covariance_xy), diff_xy[i]))
    return mahalanobisDistance

# Structural Similarity Distance
def structuralSimilarity(testImage, trainingImage):
    testImage = np.reshape(testImage, [28, 28])
    trainingImage = np.reshape(trainingImage, [28, 28])

    #  Structure similarity distance calculation
    ssimDistance = measure.compare_ssim(testImage, trainingImage, data_range=testImage.max() - trainingImage.min())
    return ssimDistance

def IDMD(m1, m2):

    # Convert image to 28 X 28
    m1 = np.reshape(m1, [28, 28])
    m2 = np.reshape(m2, [28, 28])

    # Find height and weight
    width, height = m1.shape
    # Set constants
    w0 = 1
    w1 = 1
    p = 2
    s1 = {}

    # IDMD algorithm implementation
    distance = 0
    for i1 in range(width):
        for j1 in range(height):
            i5 = 1
            for i2 in range(-w0, w0):
                for j2 in range(-w0, w0):
                    s2 = 0
                    for i3 in range(-w1, w1):
                        for j3 in range(-w1, w1):
                            v1 = m1[i1+i3][j1+j3]
                            v2 = m2[i1+i3+i2][j1+j3+j2]
                            s2 = s2 + pow(v1-v2, p)
                    s1[i5] = s2
                    i5 = i5 + 1
            distance = distance + min(s1.values())
    return distance


# Get Nearest Neighbors
def getKNearestNeighbors(trainingSet, testInstance, k, trainingLabel, distanceType):
    distances = []

    # Calculate Distance of each Neighbor
    # Euclidean Distance Loop
    if distanceType == 1:
        # Loop through each training data with test data
        for x in range(len(trainingSet)):
            # Find Euclidean Distance between training and test data
            dist = euclideanDistance(testInstance, trainingSet[x])
            distances.append((trainingSet[x], dist, trainingLabel[x]))

    # Mahalanobis Distance Loop
    elif distanceType == 2:
        # Loop through each training data with test data
        for x in range(len(trainingSet)):
            # Find Mahalanobis Distance between training and test data
            dist = mahalanobisDistance(testInstance, trainingSet[x])
            distances.append((trainingSet[x], dist, trainingLabel[x]))

    # Structural Similarity Distance Loop
    elif distanceType == 3:
        # Loop through each training data with test data
        for x in range(len(trainingSet)):
            # Find Structural Similarity Distance between training and test data
            dist = structuralSimilarity(testInstance, trainingSet[x])
            distances.append((trainingSet[x], dist, trainingLabel[x]))

    # IDMD Distance Loop
    elif distanceType == 4:
        # Loop through each training data with test data
        for x in range(len(trainingSet)):
            # Find IDMD between training and test data
            dist = IDMD(testInstance, trainingSet[x])
            # Append pixel data, distance and label
            distances.append((trainingSet[x], dist, trainingLabel[x]))

    # Sort Distance of Neighbors in ascending order
    distances.sort(key=operator.itemgetter(1))

    # Get K nearest neighbors from sorted distance list
    neighbors = []
    for x in range(k):
        # Append Distance and Label
        neighbors.append((distances[x][0], distances[x][2]))

    return neighbors


# Get a label with highest label count among nearest neighbors
def getPredictedNeighborClass(neighbors):
    classVotes = {}

    # Compute number of label occurances among nearest neighbors
    for x in range(len(neighbors)):
        label = neighbors[x][1]
        if label in classVotes:
            # Compute count of each label
            classVotes[label] += 1
        else:
            # Intialize count of label
            classVotes[label] = 1

    # Select label with maximum count
    predictedLabel = max(classVotes.items(), key=operator.itemgetter(1))[0]
    return predictedLabel


# Get Hu Moments of an image
def getHuMomentsOfImage(image):

    # Convert the image to 28 X 28
    imageData = np.reshape(image, [28, 28])

    # Get all the moments
    moments = cv2.moments(imageData, 1)
    # Get Hu Moments
    hu_moments = cv2.HuMoments(moments)
    # Convert to 1 X 7
    hu_moments = hu_moments.reshape(1, 7)[0]
    # Convert each element to float
    hu_moments = np.float32(hu_moments)

    return hu_moments


def getHuMomentsOfImageSet(imageSet):
    trainingMoments = []

    for x in range(len(imageSet)):
        hu_moments = getHuMomentsOfImage(imageSet[x])
        trainingMoments.append(hu_moments)

    return trainingMoments


def topolar(img, order=5):
    max_radius = 0.5 * np.linalg.norm(img.shape)

    def transform(coords):
        theta = 2.0 * np.pi * coords[1] / (img.shape[1] - 1.)
        radius = max_radius * coords[0] / img.shape[0]
        i = 0.5 * img.shape[0] - radius * np.sin(theta)
        j = radius * np.cos(theta) + 0.5 * img.shape[1]
        return i, j

    polar = geometric_transform(img, transform, order=order, mode='nearest', prefilter=True)
    return polar

# Get Fourier-Descriptor of the image
def findFourierDescriptor(img):

    # Reshape image to 28X28
    img = np.reshape(img, [28, 28])

    # Convert Image to Polar Co-ordinates
    img = topolar(img)

    # Finding threshold of the image
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Finding the contour of the image
    _, contour, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Finding a complex-valued vector using the coordinates obtained
    contour_array = contour[0][:, 0, :]
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)

    # Real part and Imaginary part of the contour
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]

    # Discrete Fourier transform of the complex valued vector
    dft = np.fft.fft(contour_complex)

    # Get Magnitude of the discrete fourier transform
    fourier_descriptors = abs(dft)

    return fourier_descriptors



# ---------------------- K-NN Methods ---------------------------#

#  Find accuracy of knn using raw pixels
def normal_knn(training_set, train_label, test_set, test_label, distanceType):
    # Generate predictions
    predictions = []

    for x in range(len(test_set)):

        # Rotated Test Image
        test_image = rotateImage90Degrees(test_set[x], rotationAngle)

        # Get K Nearest Neighbors of test image
        neighbors = getKNearestNeighbors(training_set, test_image, k, train_label, distanceType)

        # Predict label using nearest neighbors
        result = getPredictedNeighborClass(neighbors)

        # Adding the predicted labels in a list
        predictions.append(result)
        # Print the predicted value and label from test data
        print('predicted=' + repr(result) + ', actual=' + repr(test_label[x]))

    # Compute Accuracy using predicted labels and actual label from test data
    accuracy = getAccuracy(test_label, predictions)
    # Print Accuracy in percentage
    print('Accuracy: ' + repr(accuracy) + '%')

# Finding Accuracy of Knn using Hu Moments
def huMomentKnn(training_set, train_label, test_set, test_label, distanceType):

    # Generate predictions
    predictions = []

    # Get Hu Moments for all the training images
    trainingHuMoments = getHuMomentsOfImageSet(training_set)

    for x in range(len(test_set)):

        # Rotated Test Image
        testImage = rotateImage90Degrees(test_set[x], rotationAngle)

        # Get HuMoments of Test Image
        testHuMoments = getHuMomentsOfImage(testImage)

        # Get K Nearest Neighbors of test image
        neighbors = getKNearestNeighbors(trainingHuMoments, testHuMoments, k, train_label, distanceType)

        # Predict the label from the nearest neighbors
        result = getPredictedNeighborClass(neighbors)

        # Adding the predicted labels in a list
        predictions.append(result)
        # Print predicted label and actual label from test data
        print('predicted=' + repr(result) + ', actual=' + repr(test_label[x]))

    # Computing accuracy using predicted labels and actual labels from test data
    accuracy = getAccuracy(test_label, predictions)
    # Print Accuracy in percentage
    print('Accuracy: ' + repr(accuracy) + '%')

# Find Accuracy of Knn using fourier descriptors
def fourierDescriptorKnn(training_set, train_label, test_set, test_label, distanceType):

    # Generate predictions
    predictions = []

    # Get Training Fourier Descriptors
    trainingDescriptors = []
    for x in range(len(training_set)):
        descriptors = findFourierDescriptor(training_set[x])
        trainingDescriptors.append(descriptors)

    for x in range(len(test_set)):

        # Rotated Test Image
        testImage = rotateImage90Degrees(test_set[x], rotationAngle)

        # Get Fourier Descriptors of Test Image
        testDescriptors = findFourierDescriptor(testImage)

        # Get K Nearest Neighbors using Fourier Descriptors
        neighbors = getKNearestNeighbors(trainingDescriptors, testDescriptors, k, train_label, distanceType)

        # Predict label using nearest neighbors
        result = getPredictedNeighborClass(neighbors)

        # Add predicted value to the list
        predictions.append(result)
        # Print the predicted value and label from test data
        print('predicted=' + repr(result) + ', actual=' + repr(test_label[x]))

    # Compute Accuracy using predicted labels and actual label from test data
    accuracy = getAccuracy(test_label, predictions)
    # Print Accuracy in percentage
    print('Accuracy: ' + repr(accuracy) + '%')


def main():

    # Get training and test data sets from input files
    training_set, train_label, test_set, test_label = getInputFiles()

    # Limit the number of data
    training_set = training_set[:TRAINING_LIMIT]
    train_label = train_label[:TRAINING_LIMIT]
    test_set = test_set[:TEST_LIMIT]
    test_label = test_label[:TEST_LIMIT]

    try:
        # Select the type of Algorithm to be executed for accuracy calculation
        knnType = input("1- Knn\t2- Hu Knn\t3- Fourier Descriptor Knn\n")
    except:
        print("Invalid Input")

    if knnType == "1":
        # Select Type of distance calculation to be used
        distanceType = int(input("1- Euclidean\t2- Mahalanobis\t3- SSIM\t4- IDMD\n"))
        if (distanceType > 4 or distanceType < 1):
            print("Invalid Distance Type")
        else:
            # Knn with Raw Pixels
            normal_knn(training_set, train_label, test_set, test_label, distanceType)

    elif knnType == "2":
        # Select Type of distance calculation to be used
        distanceType = int(input("1- Euclidean\t2- Mahalanobis\n"))
        if(distanceType > 2 or distanceType < 1):
            print("Invalid Distance Type")
        else:
            # Knn with Hu Moments
            huMomentKnn(training_set, train_label, test_set, test_label, distanceType)

    elif knnType == "3":
        # Select Type of distance calculation to be used
        distanceType = int(input("1- Euclidean\t2- Mahalanobis\n"))
        if (distanceType > 2 or distanceType < 1):
            print("Invalid Distance Type")
        else:
            # Knn with Fourier Descriptors
            fourierDescriptorKnn(training_set, train_label, test_set, test_label, distanceType)
    else:
        print("Invalid input")


main()



# Can be used to Display Image
def display_image(img):
    img = np.reshape(img, [28, 28])
    cv2.imshow('Image', img)
    cv2.waitKey(0)

