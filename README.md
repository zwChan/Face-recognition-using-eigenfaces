EigenFace
=========

EigenFace implementation of the method in [Turk and Pentland's paper][1].

### Requirements
- Numpy
- Matplotlib
- cv2 (openCV2)


#### To Run
Put the training image into directory `example`, each class of images should be in the same subdir, since the name of
the subdir will be the name of the class.

Contributions:
- Part of the code is from [bugra][5];
- Great explanation from [Shubhendu Trivedi's blog][0]

  [0]: https://onionesquereality.wordpress.com/2009/02/11/face-recognition-using-eigenfaces-and-distance-classifiers-a-tutorial/
  [1]: http://www.face-rec.org/algorithms/PCA/jcn.pdf
  [2]: http://www.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf
  [3]: http://cs.gmu.edu/~kosecka/cs803/pami97.pdf
  [5]: https://github.com/bugra/EigenFace


The following is the draft of a brief report for this project

=====================
==================
===========

# Face recognition using Eigenfaces

## Abstract
This project focused on the methodology of Turk and Pentland¡¯s paper, Face recognition using eigenfaces. We implemented the workflow suing basic algebra function of Numpy, including images preprocessing, eigenfaces construction, eigenspace representation of images, face recognition based on K-nn (K near neighbors) algorithm, performance evaluation. For performance evaluation, we worked on two datasets, ATT face dataset (formerly 'The ORL Database of Faces') and cropped Yale face database B (include its extension).

## Datasets:
### AT&T "The Database of Faces" (formerly "The ORL Database of Faces")
Ten different images of each of 40 distinct subjects. The images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement)
#### The Yale Face Database B (Cropped)
Contains 2414 single light source images of 38 subjects each seen under different poses and illumination conditions. For every subject in a particular pose, only the face was extracted from the original database. The Yale face is widely considered more difficult than the AT&T database.

### Methodology, Workflow and Result
1.	Load images and convert every of them into a Numpy matrix xi;
2.	Compute the mean ?:  

Figure 1 AT&T database mean image  
![Figure 1 AT&T database mean image](/result/att_mean_image.png?raw=true "AT&T database mean image")

Figure 2 Yale database mean image
![Figure 1 Yale database mean image](/result/yale_mean_image.png?raw=true "Yale database mean image")

3.	Compute the normalized images:

4.	Compute the ¡°Covariance Matrix¡± S, which is different from the covariance matrix, in order to avoid huge matrix for eigen decomposition problem:

5.	Compute the eigenvalue and eigenvector

Figure 3 Percentage of variance for each eigenvector (AT&T database)
![result](/result/att_variance_distribution.png?raw=true)

Figure 4  Percentage of variance for each eigenvector (Yale database)
![result](/result/yale_variance_distribution.png?raw=true)

Figure 5 Top 16 eigenfaces (AT&T database)
![result](/result/att_top_16_eigenfaces.png?raw=true)

Figure 6 Top 16 eigenfaces (Yale database)
![result](/result/yale_top_16_eigenfaces.png?raw=true)

6.	Project an image into eigenspace and reconstruct using K eigenfaces:
W = vTXi  
Xf = V W  

Figure 7 Reconstruct using 80% variance (AT&T)
![result](/result/att_var080_faces43.png?raw=true)

Figure 8 Reconstruct using 90% variance (AT&T)
![result](/result/att_var090_faces110.png?raw=true)

Figure 9 Reconstruct using 95% variance (AT&T)
![result](/result/att_var095_faces189.png?raw=true)

Figure 10 Reconstruct using 99% variance (AT&T)
![result](/result/att_var099_faces324.png?raw=true)

Figure 11 Reconstruct using 80% variance (Yale)
![result](/result/yale_var080_faces4.png?raw=true)

Figure 12 Reconstruct using 90% variance (Yale)
![result](/result/yale_var090_faces22.png?raw=true)

Figure 13 Reconstruct using 95% variance (Yale)
![result](/result/yale_var095_faces62.png?raw=true)

Figure 14 Reconstruct using 99% variance (Yale)
![result](/result/yale_var090_faces189.png?raw=true)

7.	Recognition
For an image, we projected it into the eigenspace, and the image was considered as the linear combination of the eigenfaces. The weights of the corresponding eigenfaces therefore represented the image. We only used the top n eigenfaces to represent an image, where the n was determined by how much variance this sub-eigenspace can represent. We investigated 99%, 95%, 90% and 80% percent of variance for both datasets. For AT&T face dataset, the n were 324, 189, 180 and 43, respectively; for Yale face dataset, the n were 297, 62, 22 and 4, respectively.  
To recognize an unknown face, we used the Knn algorithm to find the close subject in the database. For each image in a dataset, we considered it as a query image and the other images in the dataset as training data. We got the nearest K neighbors and let them vote to determine the label of the query image. Whenever there was a tie, we used the label with the least average distance.  
If the predict label is the same with the ground label, it is a true positive; otherwise, it is a false positive. We calculated the precision as the performance of the recognition of the result.  
We also investigated different k in Knn algorithm (k from 1 to 10).

Figure 15 Precisions for different k neighbors and n percent of variances (AT&T)
![result](/result/att_precision.png?raw=true)


Figure 16 Precisions for different k neighbors and n percent of variances (Yale)
![result](/result/yale_precision.png?raw=true)


## Discussion:
The eigenfaces is one of the most popular approaches to represent an image, with the basic idea that the top k component eigenvectors (eigenfaces) represent as much variance as possible. This criterion need not to be meaningful. It is also susceptible to illumination and background around the face. Fisherfaces [6] is considered to be a better representation than eigenfaces since it is more robust to illumination. But both of them do not contain semantic meanings as human to understand a face image. A possible further study is the deep neural network approach that produce the state of the art performance by now.

## Code:
The code is available on Github:   https://github.com/zwChan/Face-recognition-using-eigenfaces

## Reference:
* Turk, Matthew A., and Alex P. Pentland. "Face recognition using eigenfaces." Computer Vision and Pattern Recognition, 1991. Proceedings CVPR'91., IEEE Computer Society Conference on. IEEE, 1991.
* Turk, Matthew, and Alex Pentland. "Eigenfaces for recognition." Journal of cognitive neuroscience 3.1 (1991): 71-86.
* Belhumeur, Peter N., Jo?o P. Hespanha, and David J. Kriegman. "Eigenfaces vs. fisherfaces: Recognition using class specific linear projection." IEEE Transactions on pattern analysis and machine intelligence 19.7 (1997): 711-720.
* http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
* http://vision.ucsd.edu/content/extended-yale-face-database-b-b
* Belhumeur, Peter N., Jo?o P. Hespanha, and David J. Kriegman. "Eigenfaces vs. fisherfaces: Recognition using class specific linear projection." IEEE Transactions on pattern analysis and machine intelligence 19.7 (1997): 711-720.
