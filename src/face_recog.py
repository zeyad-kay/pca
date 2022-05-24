import os
import cv2
import numpy as np
from sklearn import preprocessing
import math
def train_data(data_path,eigen_faces_num):
    train_images = []
    images = []
    persons = []
    for filename in os.scandir(data_path):
        if filename.is_file():
            file = os.path.basename(filename.path)
            persons.append(file[:file.find("-")])
            img = cv2.imread(filename.path,cv2.IMREAD_GRAYSCALE)
            images.append(img)
            flatten_img = img.flatten()
            train_images.append(flatten_img)
    train_images = np.array(train_images)
    mean_img = np.sum(train_images,axis=0,dtype='float64')/train_images.shape[0]
    zero_mean_train = train_images - mean_img
    cov_matrix = zero_mean_train.dot(zero_mean_train.T)/train_images.shape[0]
    eigenvalues,eigenvectors = np.linalg.eig(cov_matrix)
    indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:,indices]
    images_projection = zero_mean_train.T.dot(eigenvectors)
    eigen_faces = preprocessing.normalize(images_projection.T)
    projections = []
    for i in range(train_images.shape[0]):
        projections.append(eigen_faces[:eigen_faces_num].dot(zero_mean_train[i]))
    return projections,mean_img,eigen_faces,images,persons

def recog_face(test,projections,mean_img,eigen_faces,images,persons,eigen_faces_num,thres):
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    flatten_test = test.flatten()
    zero_mean_test = flatten_test - mean_img
    E = eigen_faces[:eigen_faces_num].dot(zero_mean_test)
    test_projected = eigen_faces[:eigen_faces_num].T.dot(E)
    diff = zero_mean_test-test_projected
    beta = math.sqrt(diff.dot(diff))
    beta = math.sqrt(diff.dot(diff))
    if beta<thres:
        face_detected = True
    else:
        face_detected = False
    smallest_dist = None 
    img_idx = 0 
    for z in range(len(projections)):
        diff = E-projections[z]
        imgs_dist = math.sqrt(diff.dot(diff))
        if smallest_dist==None:
                smallest_dist=imgs_dist
                img_idx = z
        if smallest_dist>imgs_dist:
            smallest_dist=imgs_dist
            img_idx=z
    return face_detected,images[img_idx],persons[img_idx]