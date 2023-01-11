from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# This function loads the dataset and centers it. This process isn't entirely neccessary for PCA, but makes it easier to perform PCA.
def load_and_center_dataset(filename):
    imageDataset = np.load(filename)
    centeredDataset = imageDataset - np.mean(imageDataset, axis=0)
    return centeredDataset

# This function calculates the covariance matrix of the dataset. The covariance matrix shows how data is spread across the dimensions of the dataset.
def get_covariance(dataset):
    return np.dot(np.transpose(dataset), dataset)/(len(dataset)-1)

# This function calculates the eigenvalues and eigenvectors of the covariance matrix. It returns the top m eigenvalues and eigenvectors.
def get_eig(S, m):
    Lambda = np.diag(np.flip(eigh(S, subset_by_index=[len(S)-m, len(S)-1])[0]))
    U = np.fliplr(eigh(S, subset_by_index=[len(S)-m, len(S)-1])[1])
    return Lambda, U

# This function gets all of the eigenvalues/vectors that explain more than a certain proportion of the variance. 
def get_eig_prop(S, prop):
    eigenvalueThreshold = sum(eigh(S)[0])*prop
    Lambda = np.diag(np.flip(eigh(S, subset_by_value=[eigenvalueThreshold, np.inf])[0]))
    U = np.fliplr(eigh(S, subset_by_value=[eigenvalueThreshold, np.inf])[1])       
    return Lambda, U

# Given an image in the dataset and the eigenvector from our previous functions, this function computes the PCA representation of the image.
def project_image(image, U):
    print(len(U))
    proj = 0
    for j in range(len(U[0])):
        alphaij = np.dot(image, np.transpose(U)[j])
        proj += np.dot(alphaij, np.transpose(U)[j])
    return proj

def display_image(orig, proj):
    origReshaped = np.transpose(orig.reshape([32, 32]))
    projReshaped = np.transpose(proj.reshape([32, 32]))

    fig = plt.figure(figsize=(2,1))
    fig.add_subplot(1,2,1)
    plt.title('Original')
    plt.imshow(origReshaped, aspect='equal')
    plt.colorbar()
    fig.add_subplot(1,2,2)
    plt.title('Projection')
    plt.imshow(projReshaped, aspect='equal')
    plt.colorbar()

    plt.show()
    return


x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
projection = project_image(x[0], U)
display_image(x[0], projection)