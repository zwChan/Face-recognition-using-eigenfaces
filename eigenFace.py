import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import bunch

# All images are supposed to be the same size, say, N*L
IMAGE_DIR = "C:\\fsu\class\\computer vision\\code\\term\\EigenFace\\example"


class EigenFace(object):
    # load images, and start other processing
    def __init__(self, image_path=IMAGE_DIR,variance_pct=0.99,dropTopK=0):
        # the least variance percentage we want the top K eigen vector to cover.
        self.variance_pct = variance_pct
        # don't use the top k eigen vectors
        self.dropTopK = dropTopK
        # the original images corresponding to its name
        self.image_dictionary = []
        image_names = [image for image in os.listdir(image_path) if not image.startswith('.')]
        for idx,image_name in enumerate(image_names):
            img = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE).astype(np.float64)
            if idx == 0:
                # the shape of the image. They are sopposed to be the same
                self.imgShape = img.shape
                # the normalized image matrix. it will be normalized by subtracting from the average image later
                self.vector_matrix = np.zeros((self.imgShape[0]*self.imgShape[1], len(image_names)),dtype=np.float64)
            #img = cv2.pyrDown(img)
            self.image_dictionary.append((image_name,img))
            self.vector_matrix[:,idx] = img.flatten()
        self.get_eigen()

    # use the method describing in Turk and Pentland's paper.
    def get_eigen(self):
        mean_vector = self.vector_matrix.mean(axis=1)
        for ii in range(self.vector_matrix.shape[1]):
            self.vector_matrix[:,ii] -= mean_vector
        shape = self.vector_matrix.shape
         # if there is huge number of training images. Usually go for 'else' branch.
        if (shape[0]<shape[1]):
            lamb, u = np.linalg.eig(np.dot(self.vector_matrix,self.vector_matrix.T))
            # pass
        else:
            lamb, v = np.linalg.eig(np.dot(self.vector_matrix.T, self.vector_matrix))
            u = np.dot(self.vector_matrix,v)
            # Normalizing u to ||u||=1
            norm = np.linalg.norm(u,axis=0)
            u = u / norm
            lamb = lamb * norm
        # print (lamb)
        # print (np.linalg.norm(u,axis=0))
        standard_deviation = lamb**2/float(len(lamb))
        variance_proportion = standard_deviation / np.sum(standard_deviation)
        eigen = bunch.Bunch()
        eigen.lamb = lamb
        eigen.u = u
        eigen.variance_proportion = variance_proportion
        eigen.mean_vector = mean_vector
        self.pca = eigen

    def getWeight4Training(self):
        self.K = self.get_number_of_components_to_preserve_variance(self.variance_pct)
        self.weightTraining = np.dot(self.pca.u.T, self.vector_matrix).T
        return self.weightTraining

    def get_eigen_value_distribution(self):
        data = np.cumsum(self.pca.lamb) / np.sum(self.pca.lamb)
        plt.scatter(range(data.size), data)
        return data

    def get_number_of_components_to_preserve_variance(self, variance=.95):
        for ii, eigen_value_cumsum in enumerate(self.get_eigen_value_distribution()):
            if eigen_value_cumsum >= variance:
                return ii

    def getWeight4NormalizedImg(self, imgNormlized):
        return np.dot(self.pca.u.T,imgNormlized)
    def getWeight4img(self,img):
        return self.getWeight4NormalizedImg(img.flatten-self.pca.mean_vector)




    ################   --------- show the result --------- ###################

    def plot_image_dictionary(self):
        dictionary = self.image_dictionary
        num_row_x = num_row_y = int(np.floor(np.sqrt(len(dictionary)-1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii, (name, v) in enumerate(dictionary):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(v, cmap=plt.cm.gray)
            axarr[div, rem].set_title('{}'.format(name.split(".")[0]).capitalize())
            axarr[div, rem].axis('off')
            if ii == len(dictionary) - 1:
                for jj in range(ii, num_row_x*num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')

    def plot_eigen_vector(self, n_eigen=None):
        if n_eigen is None:
            self.plot_eigen_vectors()
        else:
            plt.imshow(np.reshape(self.pca.u[:,n_eigen], self.imgShape), cmap=plt.cm.gray)

    def plot_eigen_vectors(self):
        number = self.pca.u.shape[1]
        num_row_x = num_row_y = int(np.floor(np.sqrt(number-1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii in range(number):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(np.reshape(self.pca.u[:,ii], self.imgShape), cmap=plt.cm.gray)
            axarr[div, rem].axis('off')
            if ii == number - 1:
                for jj in range(ii, num_row_x*num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')

    def get_average_weight_matrix(self):
        return np.reshape(self.pca.mean_vector, self.imgShape)

    def plot_mean_vector(self):
        fig,axarr = plt.subplots()
        axarr.set_title(" plot_mean_vector")
        axarr.imshow(self.get_average_weight_matrix(), cmap=plt.cm.gray)

    def plot_pca_components_proportions(self):
        fig,axarr = plt.subplots()
        axarr.set_title(" plot_pca_components_proportions")
        axarr.scatter(range(self.pca.variance_proportion.size), self.pca.variance_proportion)


    def plot_eigen_value_distribution(self):
        fig,axarr = plt.subplots()
        axarr.set_title(" plot_eigen_value_distribution")
        data = np.cumsum(self.pca.lamb,) / np.sum(self.pca.lamb)
        axarr.scatter(range(data.size), data)

    # plot the weights of top 2 component of all the training image
    def plotTrainingClass(self):
        fig,axarr = plt.subplots()
        axarr.set_title(" plotTrainingClass")
        ws = eigen_face.getWeight4Training()
        axarr.scatter(ws[:,0],ws[:,1])
        for idx in range(0,ws.shape[0]):
            name = eigen_face.image_dictionary[idx][0].split("_")[0]
            axarr.text(ws[idx,0],ws[idx,1],name)


eigen_face = EigenFace(variance_pct=0.95,dropTopK=0)
eigen_face.plot_image_dictionary()
eigen_face.plot_eigen_vector()
eigen_face.plot_mean_vector()
eigen_face.plot_pca_components_proportions()
eigen_face.plot_eigen_value_distribution()
eigen_face.plotTrainingClass()

plt.show()
