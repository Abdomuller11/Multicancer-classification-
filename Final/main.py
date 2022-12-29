import random

import cv2 as cv
import numpy as np
from glob import glob
import argparse
#from helpers import *
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
from sklearn.datasets import make_blobs
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
import imutils                      #basic image processing functions
from sklearn import *       #mach learn
from keras import *
from tqdm import tqdm      #create Progress Bars.
import itertools             #iterate over data structures
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,classification_report

import time



# import timeit
#
# start = timeit.default_timer()













###########






##########

###other classes

class FileHelpers:

    def __init__(self):
        pass

    # def getFiles(self, path):
    #     """
	# 	- returns  a dictionary of all files
	# 	having key => value as  objectname => image path
    #
	# 	- returns total number of files.
    #
	# 	"""
    #     imlist = {}
    #     count = 0
    #     for each in os.listdir(path):
    #         print(" #### Reading image category ", each, " ##### ")
    #         imlist[each] = []
    #         for imagefile in os.listdir(path + '/' + each):
    #             print("Reading file ", imagefile)
    #             im = cv.imread(path + '/' + each + '/' + imagefile, 0)
    #             imlist[each].append(im)  # accordian:img0....
    #             count += 1
    #     print("imlist= ", imlist)
    #     return [imlist, count]  ##imlist=dict accordian:img0-1-2 اللي جواه ....... , count

    def preparation(self,x, y, labels):
        imlist = {}
        count = 0
        for key, val in labels.items():
            imlist[val] = []
            for k, j in zip(x, y):
                if (j == key):
                    imlist[val].append(k)
                    count += 1

        return [imlist, count]



class BOVHelpers:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans_obj = KMeans(n_clusters=n_clusters)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.clf = SVC()

    # n_samples = 1000
    # n_features = 5;
    # n_clusters = 3;
    #
    # # aint this sweet
    # X, y = make_blobs(n_samples, n_features)
    # # X => array of shape [nsamples,nfeatures] ;;; y => array of shape[nsamples]
    #
    # # X : generated samples, y : integer labels for cluster membership of each sample
    # #
    # #
    #
    # # performing KMeans clustering
    #
    # ret = KMeans(n_clusters=n_clusters).fit_predict(X)
    # #print(ret)
    #
    # __, ax = plt.subplots(2)
    # ax[0].scatter(X[:, 0], X[:, 1])
    # ax[0].set_title("Initial Scatter Distribution")
    # ax[1].scatter(X[:, 0], X[:, 1], c=ret)
    # ax[1].set_title("Colored Partition denoting Clusters")
    # plt.scatter
    #plt.show()



    def formatND(self, l):
        """
		restructures list into vstack array of shape
		M samples x N features for sklearn

		"""
        vStack = np.array(l[0])
        for remaining in l[1:]:
            vStack = np.vstack((vStack, remaining)) ##vstack contatenator
        self.descriptor_vstack = vStack.copy()
        return

    def cluster(self):
        # des =وصف
        """
        cluster using KMeans algorithm,
            kmeans op on des_vstack
            fit  100center randomly                         100defualt   100*128des

            predict تنتمي لانهي كلاستر   keypoint

        """
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)  # 1500*1    600+400+500    keypoints*1

    def developVocabulary(self, n_images, descriptor_list):

        """
        Each cluster denotes a particular visual word
        Every image can be represeted as a combination of multiple
        visual words. The best method is to generate a sparse histogram
        that contains the frequency of occurence of each visual word

        Thus the vocabulary comprises of a set of histograms of encompassing
        all descriptions for all images

        """

        self.mega_histogram = np.array(
            [np.zeros(self.n_clusters) for i in range(n_images)])  # عدد الصور في عدد الكلاسترز
        old_count = 0
        for i in range(n_images):  # rows
            l = len(descriptor_list[i])  # لكل صورة عشان افرق بين الصور
            for j in range(l):  # j in 600
                idx = self.kmeans_ret[old_count + j]  # 20
                self.mega_histogram[i][idx] += 1  # +1 in histogram
            old_count += l  # وقفت فين اخر مرة
        print("Vocabulary Histogram Generated")

    def plotHist(self, vocabulary=None):
        print("Plotting histogram")
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:, h], dtype=np.int32)) for h in range(self.n_clusters)])  ##sum sam column

        print(y_scalar)

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()

    def standardize(self, std=None):
        """

		standardize is required to (normalize) the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.
            svm gradiant descent
		"""
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)  # obj     mean and std deviation
            self.mega_histogram = self.scale.transform(self.mega_histogram)  # norm -mean/std dev
        else:
            print("STD not none. External STD supplied")
            self.mega_histogram = std.transform(self.mega_histogram)





    def train(self, train_labels):
        """
		uses sklearn.svm.SVC classifier (SVM)


		"""
        print("Training SVM")  #Training SVM
        # print(self.clf)
        # print("Train labels", train_labels)
        self.clf.fit(self.mega_histogram, train_labels)#mega =feature extraction
        print("Training completed")






class ImageHelpers:
    def __init__(self):
        #self.sift_object = cv.xfeatures2d.SIFT_create()

        self.sift_object = cv.SIFT_create()

    # def gray(self, image):
    #     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #     return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]






class BOV:
    def __init__(self, no_clusters,x_train,y_train,labels0,x_test,y_test,labels1):
        self.no_clusters = no_clusters
        # self.train_path = None
        # self.test_path = None
        self.im_helper = ImageHelpers()  ##create obj
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []
        self.x0 = x_train
        self.y0 = y_train
        self.labels0 = labels0   ##training
        self.x1=x_test
        self.y1=y_test
        self.labels1=labels1     #testing

    def trainModel(self):
        """
        This method contains the entire module
        required for training the bag of visual words model(features)

        Use of helper functions will be extensive.

        """

        # read file. prepare file lists.
        path=0
        what_img=cv.imread(r"{}".format(path),0)

        brain_match = cv.imread(r"C:\Users\Administrator\Desktop\PY\CV2023CSYSDataset\Brain scans\No_tumor\Train\no3.jpg", 0)
        breast_match = cv.imread(r"C:\Users\Administrator\Desktop\PY\CV2023CSYSDataset\Breast scans\malignant\Train\malignant (5).png", 0)
        what_img= cv.threshold(what_img, 100, 255, cv.THRESH_TRUNC)[1]
        thresh1 = cv.threshold(brain_match, 100, 255, cv.THRESH_TRUNC)[1]
        thresh2 = cv.threshold(breast_match, 100, 255, cv.THRESH_TRUNC)[1]

        #sift_object = cv.SIFT_create()

        orb = cv.ORB_create()
        # keypoints1, descriptors1 = sift_object.detectAndCompute(brain_match, None)
        # keypoints2, descriptors2 = sift_object.detectAndCompute(breast_match, None)
        kp0,des0=orb.detectAndCompute(what_img, None)
        keypoints1, descriptors1 = orb.detectAndCompute(thresh1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(thresh2, None)

        matcher = cv.BFMatcher(cv.NORM_HAMMING)
        match0 = matcher.knnMatch(des0, descriptors1, k=2)
        match1 = matcher.knnMatch(des0, descriptors2, k=2)
        good_brain = []
        good_breast = []

        for m, n in match0:
            if m.distance < 0.8 * n.distance:
                good_brain.append([m])
        for m, n in match1:
            if m.distance < 0.8 * n.distance:
                good_brain.append([m])

        #final_img = cv.drawMatchesKnn(brain_match, keypoints1, brain_match, keypoints2, match0, None)
        # plt.imshow(final_img)
        # plt.show()
        # print(len(goo_brain))
        if len(good_brain)>len(good_breast):
            print("this img is brain")
        else:
            print("this img is breast")







        self.images, self.trainImageCount = self.file_helper.preparation(self.x0,self.y0,self.labels0)  ##train count=57 صورة في الarr
        # extract SIFT Features from each image
        label_count = 0
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word  # accordian 0
            print("Computing Features for ", word)
            for im in imlist:
                # cv2.imshow("im", im)
                # cv2.waitKey()
                self.train_labels = np.append(self.train_labels, label_count)  # img0  0
                kp, des = self.im_helper.features(im)  # detect and compute    return  keypoints+descriptors
                self.descriptor_list.append(des)  # des for img0....  600*128, 400*128, 500*128     len=57         فعاوز افرضه   1500*128
            ## ليه عشان kmeans المفروض ميرفرقش بين الصور وياخد   اللي شبه بعض واحطه في كلاستر لذلك يجب الفرد اولا

            label_count += 1  # accordian 0, dollar 1,.....

        # perform clustering
        self.bov_helper.formatND(self.descriptor_list)
        self.bov_helper.cluster()  # c
        self.bov_helper.developVocabulary(n_images=self.trainImageCount, descriptor_list=self.descriptor_list)

        # show vocabulary trained
        #self.bov_helper.plotHist()

        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels)

    def recognize(self, test_img, test_image_path=None):  # علي الصورة

        """
        This method recognizes a single image
        It can be utilized individually as well.


        """

        kp, des = self.im_helper.features(test_img)  # sift features
        # print kp
        #print(des.shape)

        # generate vocab for test image  #vocab =histogram
        vocab = np.array([[0 for i in range(self.no_clusters)]])  # np.zeros
        vocab = np.array(vocab, 'float32')
        # locate nearest clusters for each of
        # the visual word (feature) present in the image

        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)  # predict علطول   #kmeans_obj لازم يبقي متشاف في اي حتة عشان هشتغل عليه    predict(des)
        # print test_ret

        # print vocab histo
        for each in test_ret:  # each=20 20 5 .....
            vocab[0][each] += 1  # only one row 1*100(1*clusters)
            # if more imgs vocab[i]

        # print (vocab)

        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)  # norm preprocess         طالما اعملت في اtrain يبقي زيها ال test
        # predict the class of the image
        lb = self.bov_helper.clf.predict(vocab)  # هل تبع 0 و1 accordian ...
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb

    # بعد كدا هقارن اللي label ب actual label
    # then calc acc.                all incorrect/total

    def ensure(self,img):

        X = img.shape[0]
        copy = np.copy(img)
        copy=cv.cvtColor(copy,cv.COLOR_RGB2GRAY)
        blur = cv.GaussianBlur(copy, (5, 5), 2)
        enh = cv.add(copy, (cv.add(blur, -100)))
        median = cv.medianBlur(enh, 7)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        gradient = cv.morphologyEx(median, cv.MORPH_GRADIENT, kernel)
        enh2 = cv.add(median, gradient)
        t = np.percentile(enh2, 85)  # 85
        ret, th = cv.threshold(enh2, t, 255, cv.THRESH_BINARY)  # THRESH_BINARY
        kernel_c = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((5 * X) / 100), int((5 * X) / 100)))  #
        kernel_e = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((3 * X) / 100), int((3 * X) / 100)))  #
        ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((7 * X) / 100), int((7 * X) / 100)))
        opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel_e)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel_c)
        erosion = cv.erode(closing, kernel_e, iterations=1)
        dilation = cv.dilate(erosion, kernel_e, iterations=1)
        masked = cv.bitwise_and(copy, copy, mask=dilation)
        s_erosion = cv.erode(masked, kernel, iterations=1)
        final = cv.morphologyEx(s_erosion, cv.MORPH_OPEN, ker)
        blur3 = cv.GaussianBlur(final, (3, 3), 0)
        enh3 = cv.add(final, (cv.add(blur3, -100)))
        upper = np.percentile(enh3, 85)  # 92
        res = cv.inRange(enh3, 0, upper)
        fin = cv.morphologyEx(res, cv.MORPH_CLOSE,
                              cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((7 * X) / 100), int((7 * X) / 100))))

        copy_rgb = cv.cvtColor(copy,
                               cv.COLOR_BGR2RGB)  # necessary step in order to print out the original image colors correctly

        contours, hierarchy = cv.findContours(fin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) > 1:
            cnt = contours[1]
            if len(contours) > 2:
                pass
            else:
                pass
            area = int(cv.contourArea(cnt))
            perimeter = int(cv.arcLength(cnt, True))
            return area
        else:
            area=0
            return area

    def testModel(self,display_max):
        """
        This method is to test the trained classifier

        read all images from testing path
        use BOVHelpers.predict() function to obtain classes of each image

        """
        correctClassifications = 0
        self.testImages, self.testImageCount = self.file_helper.preparation(self.x1,self.y1,self.labels1)

        predictions = []
        liofpredict=[]
        for word, imlist in self.testImages.items(): #هجيب الصور تيست
            print ("processing " ,word)
            for im in imlist:
                # print imlist[0].shape, imlist[1].shape
                #print (im.shape)
                cl = self.recognize(im)
                # plt.imshow(im)
                # #plt.show()
                # print("cl[0]", cl[0])
                # print(cl)
                # print("before", self.name_dict[str(int(cl[0]))])
                # print("pro",self.ensure(im))
                # if (self.ensure(im)>0 and cl==[0]):
                #     cl=[1]
                liofpredict.append(cl)
                # print ("cl:",cl)
                # print("after",self.name_dict[str(int(cl[0]))])
                predictions.append({
                    'image':im,
                    'class':cl,
                    'object_name':self.name_dict[str(int(cl[0]))],
                    'Fact':word
                })

                if(self.name_dict[str(int(cl[0]))]==word):  #لو صح
                    correctClassifications = correctClassifications + 1

        print("Test Accuracy = " + str((correctClassifications/self.testImageCount) * 100))


        #print (predictions) in order
        # for each in predictions:
        #     # cv2.imshow(each['object_name'], each['image'])
        #     # cv2.waitKey()
        #     # cv2.destroyWindow(each['object_name'])
        #     #
        #     #plt.imshow(cv.cvtColor(each['image'], cv.COLOR_GRAY2RGB))
        #     plt.imshow(each['image'])
        #     plt.title(each['object_name'])
        #     plt.show()

        #print (predictions) randomly
        display_count=0
        randomly_pred=predictions
        random.shuffle(randomly_pred)
        for each in randomly_pred:
            if(display_count<display_max):
                plt.imshow(each['image'])
                plt.suptitle(f"Fact: {each['Fact']}")
                plt.title(f"Predict: {each['object_name']}")
                if(each['Fact']==each['object_name']):
                    plt.title(
                        "SUCCESS",fontsize='small',loc='right', color= 'blue', fontweight='bold'
                    )
                plt.show()
                display_count+=1

        # cm = confusion_matrix(self.y1,li)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # ax=disp.plot()
        # plt.suptitle('Confusion Matrix');
        # ax.xaxis.set_ticklabels(['No_tumor', 'Tumor'])
        # ax.yaxis.set_ticklabels(['Tumor','No_tumor'])
        # plt.show()

        cm = confusion_matrix(self.y1,liofpredict)
        print("Brain MRI Classification :",cm)
        cmd = ConfusionMatrixDisplay(cm, display_labels=['No_tumor', 'Tumor'])
        cmd.plot()
        plt.show()

        print("Accuracy: " + str(accuracy_score(self.y1,liofpredict)))
        print('\n')
        print(classification_report(self.y1,liofpredict))












        # labels = ['No_tumor', 'Tumor']
        # cm = confusion_matrix(self.y1,liofpredict)
        # print(cm)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(cm)
        # plt.title('Confusion matrix of the classifier')
        # fig.colorbar(cax)
        # ax.set_xticklabels([''] + labels)
        # ax.set_yticklabels([''] + labels)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.show()
    # def print_vars(self):
    #     pass



































#########
print("Start Prediction Program")
print('\n')
print("Fisrt Brain Scan:(Tumor, No_tumor) ")
print('\n')
brain_set_path= r"W:/AinShams uni/Project/py & vision best/PY/CV2023CSYSDataset/Brain scans/"
breast_set_path=r"W:/AinShams uni/Project/py & vision best/PY/CV2023CSYSDataset/Breast scans/"
tumor=f"{brain_set_path}Tumor/"
notumor=f"{brain_set_path}No_tumor/"
normal=f"{breast_set_path}normal/"
benign=f"{breast_set_path}benign/"
malignant=f"{breast_set_path}malignant/"


#img preprocessing & preparation and visualization:





img = cv.imread(f"{tumor}Train/y184.jpg",0)  #261

plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("Original Image")
plt.show()

X = img.shape[0] # image's height is used to scale the structuring elements utilized in morphology operations

copy = np.copy(img) # it's a good practice to work on a copy rather than on the original image itself


##Original intensity histogram

hist, bins = np.histogram(copy.flatten(), 256, [0, 256])
plt.stem(hist, use_line_collection=True)
plt.show()

#
#
# t = np.percentile(copy,85)
# ret,th = cv.threshold(copy, t, 255, cv.THRESH_TOZERO)#THRESH_BINARY
# plt.imshow(th, cmap='gray', vmin=0, vmax=255),plt.title("FIRST THRESHOLDING")
# plt.show()
#
# print(th)
#
# blur = cv.GaussianBlur(th,(5,5),2)
# enh = cv.add(th,(cv.add(blur,-100))) # in order to avoid over-brightness of the image, blurred intensities are reduced
# plt.imshow(enh, cmap='gray', vmin=0, vmax=255),plt.title("FIRST ENHANCEMENT")
# plt.show()
#

#First enhancement

blur = cv.GaussianBlur(copy,(5,5),2)  #copy
enh = cv.add(copy,(cv.add(blur,-100))) # in order to avoid over-brightness of the image, blurred intensities are reduced -100
plt.imshow(enh, cmap='gray', vmin=0, vmax=255),plt.title("FIRST ENHANCEMENT")
plt.show()



#Intensity histogram after the first enhancement


hist2, bins2 = np.histogram(enh.flatten(),256,[0,256])
plt.stem(hist2, use_line_collection=True)
plt.show()



#Denoising

median = cv.medianBlur(img,7)#img
plt.imshow(median, cmap='gray', vmin=0, vmax=255),plt.title("DENOISED")
plt.show()




#Morphological Gradient



kernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
gradient = cv.morphologyEx(median, cv.MORPH_GRADIENT, kernel)
plt.imshow(gradient, cmap='gray', vmin=0, vmax=255),plt.title("MORPHOLOGICAL GRADIENT")
plt.show()




enh2 = cv.add(median,gradient)
plt.imshow(enh2, cmap='gray', vmin=0, vmax=255),plt.title("SECOND ENHANCEMENT")
plt.show()




hist3, bins3 = np.histogram(enh2.flatten(),256,[0,256])
plt.stem(hist3, use_line_collection=True)
plt.show()




#thre

t = np.percentile(enh2,85)#85
ret,th = cv.threshold(enh2, t, 255, cv.THRESH_BINARY)#THRESH_BINARY
plt.imshow(th, cmap='gray', vmin=0, vmax=255),plt.title("FIRST THRESHOLDING")
plt.show()



##Morphology operations
kernel_c = cv.getStructuringElement(cv.MORPH_ELLIPSE,(int((5*X)/100),int((5*X)/100))) #
kernel_e = cv.getStructuringElement(cv.MORPH_ELLIPSE,(int((3*X)/100),int((3*X)/100))) #
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE,(int((7*X)/100),int((7*X)/100)))

plt.figure(figsize=(10,10),constrained_layout = True)

opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel_e) # to eliminate small uninteresting structures th
plt.subplot(221),plt.imshow(opening, cmap='gray', vmin=0, vmax=255),plt.title("1. FIRST OPENING")

closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel_c) # to merge the remaining structures that may have been divided
plt.subplot(222),plt.imshow(closing, cmap='gray', vmin=0, vmax=255),plt.title("2. CLOSING")

erosion = cv.erode(closing,kernel_e,iterations = 1)
plt.subplot(223),plt.imshow(erosion, cmap='gray', vmin=0, vmax=255),plt.title("3. FIRST EROSION")

dilation = cv.dilate(erosion,kernel_e,iterations = 1)
plt.subplot(224),plt.imshow(dilation, cmap='gray', vmin=0, vmax=255),plt.title("4. DILATION")


plt.show()






#masking



masked = cv.bitwise_and(copy, copy, mask=dilation)#dilationclosing
plt.imshow(masked, cmap='gray', vmin=0, vmax=255),plt.title("MASKED")
plt.show()



#Second round of morphology operations


s_erosion = cv.erode(masked,kernel,iterations = 1)
plt.subplot(121),plt.imshow(s_erosion, cmap='gray', vmin=0, vmax=255),plt.title("1. SECOND EROSION")

final = cv.morphologyEx(s_erosion, cv.MORPH_OPEN, ker)
plt.subplot(122),plt.imshow(final, cmap='gray', vmin=0, vmax=255),plt.title("2. SECOND OPENING")
plt.show()


#enhance

blur3 = cv.GaussianBlur(final,(3,3),0)
enh3 = cv.add(final,(cv.add(blur3,-100)))
plt.imshow(enh3, cmap='gray', vmin=0, vmax=255),plt.title("THIRD ENHANCEMENT")
plt.show()




upper = np.percentile(enh3,85)#92
res = cv.inRange(enh3, 0, upper)
plt.imshow(res, cmap='gray', vmin=0, vmax=255),plt.title("SECOND THRESHOLDING")
plt.show()

fin = cv.morphologyEx(res, cv.MORPH_CLOSE,
                      cv.getStructuringElement(cv.MORPH_ELLIPSE, (int((7 * X) / 100), int((7 * X) / 100))))
plt.imshow(fin, cmap='gray', vmin=0, vmax=255), plt.title("LAST CLOSING")
plt.show()

copy_rgb = cv.cvtColor(copy,
                       cv.COLOR_BGR2RGB)  # necessary step in order to print out the original image colors correctly

contours, hierarchy = cv.findContours(fin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

if len(contours) > 1:
    cnt = contours[1]
    if len(contours) > 2:
        cv.drawContours(copy_rgb, contours, 2, (0, 255, 0), 3)
        plt.imshow(copy_rgb), plt.title("DETECTED TUMOR")
        plt.show()
    else:
        cv.drawContours(copy_rgb, contours, 1, (0, 255, 0), 3)
        plt.imshow(copy_rgb), plt.title("DETECTED TUMOR")
        plt.show()

    area = int(cv.contourArea(cnt))
    perimeter = int(cv.arcLength(cnt, True))

    print("Area:", area, "px")
    print("Perimeter:", perimeter, "px")
else:
    print("No tumor detected")


def load_data(dir_path,trainORtest="Train",img_size=(600,600)):   #loading data and its label(0notumor    1tumor)
    """
    Load resized images as np.arrays to workspace
    """
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path + '/' + trainORtest.title()):
                if not file.startswith('.'):
                    img = cv.imread(dir_path + path + '/' + trainORtest.title() + "/" + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)#0-->no tumor    1-->tumor    as labels
    print(f'{len(X)} images loaded from {dir_path + trainORtest.title() }  directory.')
    return X, y, labels


def plot_samples(x, y, labels_dict, n=50):
    """
    Creates a gridplot for desired number of images (n) from the specified set
    """
    for index in range(len(labels_dict)):
        imgs = x[np.argwhere(y == index)][:n]
        j = 10
        i = int(n/j)

        plt.figure(figsize=(15,6))
        c = 1
        for img in imgs:
            plt.subplot(i,j,c)
            plt.imshow(img[0])

            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.suptitle('BrainScan: {}'.format(labels_dict[index]))
        plt.show()


def crop_imgs_enhance(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    set_new = []
    for img in set_name:
        img = cv.resize(
            img,
            (190, 170),#(190, 170), #180
            interpolation=cv.INTER_CUBIC
        )
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = cv.GaussianBlur(gray, (3, 3), 2)
        gray=cv.medianBlur(gray,5)

        #blur = cv.GaussianBlur(gray, (5, 5), 2)
        #enh = cv.add(copy, (cv.add(blur, -100)))
        #median = cv.medianBlur(img, 7)

        gradient = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
        enh2 = cv.add(gray, gradient)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv.threshold(enh2,  4, 255,cv.THRESH_BINARY)[1]#gray  #10-->73.0  5-->77.0   4-->81.0  THRESH_BINARY
        #thresh=cv.adaptiveThreshold(gray,100,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,145,8)   #adaptive not global
        thresh = cv.erode(thresh, None, iterations=2)
        thresh = cv.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)
        #set not highlighted
    return np.array(set_new)


#saved folder(not necessary in project)
# def save_new_images(x_set, y_set, folder_name):
#     i = 0
#     for (img, imclass) in zip(x_set, y_set):
#         if imclass == 0:
#             cv.imwrite(folder_name+'NO/'+str(i)+'.jpg', img)
#         else:
#             cv.imwrite(folder_name+'YES/'+str(i)+'.jpg', img)
#         i += 1
#
# save_new_images(X_train_crop, y_train, folder_name='Train_crop/')
# save_new_images(X_test_crop, y_test, folder_name='Test_crop/')
#



#loading data
x_train,y_train,labels=load_data(brain_set_path,"Train")
x_test,y_test,_=load_data(brain_set_path,"Test")

y1=dict()        #for count and distribution
y1[0]=[]


y2=dict()        #for count and distribution
y2[0]=[]

for i in y_train:

    y1[0].append(np.sum(i == 0))

for ii in y_test:

    y2[0].append(np.sum(ii == 0))


train_notumor=0;train_tumor=0;test_notumor=0;test_tumor=0
for count in y1[0]:
    if count==0:
        train_notumor+=1
    else:
        train_tumor+=1

for count in y2[0]:
    if count==0:
        test_notumor+=1
    else:
        test_tumor+=1

# x-coordinates of left sides of bars
left = [1, 3, 6, 8]

# heights of bars
height = [train_notumor, train_tumor, test_notumor, test_tumor]

# labels for bars
tick_label = ['Train(NoTumor)', 'Train(Tumor)', 'Test(NoTumor)', 'Test(Tumor)']

# plotting a bar chart
plt.bar(left, height, tick_label=tick_label,
        width=1, color=['red', 'green'])

# naming the x-axis
plt.xlabel('DataSet')
# naming the y-axis
plt.ylabel('Count')
# plot title
plt.title('Statistics')

# function to show the plot
plt.show()


plot_samples(x_train, y_train, labels, 30)


#example
img=cv.imread(r"W:/AinShams uni/study/vision/PY/CV2023CSYSDataset/Brain scans/Tumor/Train/y194.jpg")  #194 good at highlight

# img = cv.resize(
#             img,
#             (200,200),
#             interpolation=cv.INTER_CUBIC
#         )
# print("h=",img.shape[0])
# print("w=",img.shape[1])

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv.threshold(gray, 45, 255, cv.THRESH_TOZERO)[1]
thresh = cv.erode(thresh, None, iterations=2)
thresh = cv.dilate(thresh, None, iterations=2)

# find contours in thresholded image, then grab the largest one
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv.contourArea)

# find the extreme points
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])
#visualization
# add contour on the image
img_cnt = cv.drawContours(img.copy(), [c], -1, (0, 255, 255), 12)#line

# add extreme points
img_pnt = cv.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
img_pnt = cv.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
img_pnt = cv.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
img_pnt = cv.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

# crop
ADD_PIXELS = 0
new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
#plotting highlighted cropped result
plt.figure(figsize=(15,6))
plt.subplot(141)
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.title('Step 1. Get the original image')
plt.subplot(142)
plt.imshow(img_cnt)
plt.xticks([])
plt.yticks([])
plt.title('Step 2. Find the biggest contour')
plt.subplot(143)
plt.imshow(img_pnt)
plt.xticks([])
plt.yticks([])
plt.title('Step 3. Find the extreme points')
plt.subplot(144)
plt.imshow(new_img)
plt.xticks([])
plt.yticks([])
plt.title('Step 4. Crop the image')
plt.show()


# apply this for each set
X_train_crop = crop_imgs_enhance(set_name=x_train)
X_test_crop = crop_imgs_enhance(set_name=x_test)



#after cropping
plot_samples(X_train_crop, y_train, labels, 30)


edges_of_image = cv.Canny(img, 100, 200)
plt.imshow(edges_of_image)
plt.title('Canny edge detector')
plt.show()


lap=cv.Laplacian(edges_of_image,cv.CV_64F,ksize=3) #second dervative edge detection
lap=np.uint8(np.absolute(lap))
plt.imshow(lap)
plt.title('laplacian filter edge detector')
plt.show()


#fast corner
fast=cv.FastFeatureDetector_create(0)
kp=fast.detect(img,None)
img2=cv.drawKeypoints(img,kp,None,color=(255,0,0))
plt.imshow(img2)
plt.title('fast feature detector')
plt.show()


#error404
# X_train_crop=cv.Canny(X_train_crop, 100, 200)
# plot_samples(X_train_crop, y_train, labels, 30)

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )


#tumor hole
template = cv.imread(r"C:\Users\Administrator\Desktop\Proj\hole2.png",0)
temp=cv.GaussianBlur(template, (5, 5), 0)
gray_match = gray.copy()
cv.imshow("Tumor",temp)
cv.waitKey(0)
w, h = temp.shape[::-1]
res = cv.matchTemplate(gray_match,temp,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(gray_match, pt, (pt[0] + w, pt[1] + h), (0,0,255), 4)

cv.imshow("detected: ",gray_match)
cv.waitKey(0)

#All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
meth=['cv.TM_CCOEFF']

for meth in meth:
    gray = gray_match.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(gray,temp,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 20, 4)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(gray,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
#
# cv.imshow("Lighting Tumor",gray)
# cv.waitKey(0)

X_train_crop_list=[]
X_test_crop_list=[]
for i in X_train_crop:
    X_train_crop_list.append(i)
for j in X_test_crop:
    X_test_crop_list.append(j)

#
# def preparation(x, y, labels):
#     imlist = {}
#     count=0
#     for key,val in labels.items():
#         imlist[val]=[]
#         for k,j in zip(x,y):
#             if (j==key):
#                 imlist[val].append(k)
#                 count+=1
#     return [imlist, count]
#
#
#
#
# imlist,count=preparation(X_train_crop_list,y_train,labels)





start_time = time.time()

bov = BOV(no_clusters=20,x_train=X_train_crop_list,y_train=y_train,labels0=labels,x_test=X_test_crop_list,y_test=y_test,labels1=_)
#20 best    84.5

bov.trainModel()

# test model and acc
bov.testModel(14)




print("first:--- %s seconds ---" % (time.time() - start_time))

# stop = timeit.default_timer()
#
# print('Time: ', stop - start)


print("###########################################################################")
print("Second: Breast Scans (Normal, Malignant, Benign)")
print('\n')



import cv2
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os
from skimage.transform import resize
from sklearn.svm import LinearSVC
from skimage import feature
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from PIL import Image, ImageEnhance
from scipy.signal import convolve2d


class ImagesExtraction: #extract features from the image using Hog algorithm
    def __init__(self):
       pass

    def features(self, image):
        # resizing image
        resized_image = resize(image, (128, 64))
        feature_des,image_hog = feature.hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)#Extract features using hog built in function
        return feature_des





class FileHelpers:

    def __init__(self):
        pass

    def getFiles(self, path):

        List_of_iages = {}
        counter = 0
        for Each_image in os.listdir(path):
            print(" #### Reading image category ", Each_image, " ##### ")
            List_of_iages[Each_image] = []
            for imagefile in os.listdir(path + '/' + Each_image):
                print("Reading file ", imagefile)
                image = cv2.imread(path + '/' + Each_image + '/' + imagefile, 0)
                List_of_iages[Each_image].append(image)
                counter += 1

        return [List_of_iages, counter]






class Classification:
    def __init__(self,way,train_path=r'C:/Users/Administrator/Desktop/train',test_path=r'C:/Users/Administrator/Desktop/test',train_path2=r'C:/Users/Administrator/Desktop/train2',test_path2=r'C:/Users/Administrator/Desktop/test2'):
        self.train_path = train_path
        self.test_path = test_path
        self.train_path2 = train_path2
        self.test_path2 = test_path2
        self.im_helper = ImagesExtraction()
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list_images = []
        self.labels = []
        self.svm_model = LinearSVC(random_state=42, tol=1e-5)
        self.way=way

    def multi_convolver(self,image, kernel, iterations):
        for i in range(iterations):
            image = convolve2d(image, kernel, 'same', boundary='fill',
                               fillvalue=0)

    def convolver_rgb(self,image, kernel, iterations=1):
        convolved_image_r = self.multi_convolver(image[:, :, 0], kernel,
                                            iterations)
        convolved_image_g = self.multi_convolver(image[:, :, 1], kernel,
                                            iterations)
        convolved_image_b = self.multi_convolver(image[:, :, 2], kernel,
                                            iterations)

        reformed_image = np.dstack((np.rint(abs(convolved_image_r)),
                                    np.rint(abs(convolved_image_g)),
                                    np.rint(abs(convolved_image_b)))) /255

        return np.array(reformed_image).astype(np.uint8)


    def trainModel(self):
        # read file. prepare file lists.
        if (self.way==1):
            self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path2)
        else:
            self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
        gaussian = (1 / 16.0) * np.array([[1., 2., 1.],
                                          [2., 4., 2.],
                                          [1., 2., 1.]])
        sharpen = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

        label_count = 0
        for File_name, List_of_images in self.images.items():
            self.name_dict[str(label_count)] = File_name
            print("Computing Features for ", File_name)
            for image in List_of_images:
                self.train_labels = np.append(self.train_labels, label_count)
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  already grayscale
                #image2=image.copy()
                #image2=cv2.medianBlur(image2,9)

                #image = cv2.threshold(image, 0, 40, cv2.THRESH_BINARY)[1]
                #enhancer = ImageEnhance.Sharpness(image)
                #image=enhancer.enhance(2)
                # kernel = np.array([[1, 1, 1], [1, 8, 1], [1, 1, 1]])
                # image = cv2.filter2D(image, -1, kernel)
                # image=self.multi_convolver(image,gaussian,2)
                #image = self.convolver_rgb(image, sharpen, 1)

                #cv2.GaussianBlur(frame, image, cv::Size(0, 0), 3);
                #image=cv2.addWeighted(image2, 1.5, image, -0.5, 0, image)
                plt.imshow(image)
                #plt.show()
                #image=cv2.GaussianBlur(image,(1,1),1)
                feature_des_image = self.im_helper.features(image)
                self.descriptor_list_images.append(feature_des_image)
                self.labels.append(label_count)

            label_count += 1
        self.descriptor_list_images = np.array(self.descriptor_list_images)
        self.labels = np.array(self.labels)
        # train Linear SVC
        print('Training on train images...')
        self.svm_model.fit(self.descriptor_list_images, self.labels)

    def testModel(self,display_max):
        #1way    mal diff from benign     0
        #2way    mal +benign as atumor    1
        if (self.way == 1):
            self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path2)
        else:
            self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)

        #self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)

        totalcorrection=0
        predictions = []  # viewing
        imgs = []
        fact = []
        for File_name, imlist in self.testImages.items():
            correctClassifications=0

            count=0
            List_of_predictions = []

            print("processing ", File_name)
            for image in imlist:
                feature_des_image = self.im_helper.features(image)
                #pred.append(self.svm_model.predict(feature_des_image))
                List_of_predictions.append(feature_des_image)
                count+=1
                imgs.append(image)
                fact.append(File_name)
            Result_of_predictions= self.svm_model.predict(List_of_predictions)
            print("The list of predictions for images :.....")
            print(Result_of_predictions)
            for i in Result_of_predictions:
                predictions.append(i)
            Brainscan_counter = 0
            Breastscan_counter = 0
            #0-->benign 1-->mal  2-->norm
            for i in Result_of_predictions:
               # n=str(int(Result_of_predictions[i]))
                if(self.name_dict[str(i)]==File_name):
                  correctClassifications = correctClassifications + 1
                  totalcorrection=totalcorrection+1
            print("Test Accuracy of " + File_name + "=" + str((correctClassifications / count) * 100))
        final=[]
        print(len(imgs))
        print(len(predictions))
        print(len(fact))
        for (j,k,fa) in zip(imgs,predictions,fact):
            if (self.way==1):
                if (k == 1):
                    label = "tumor"
                else:
                    label = "normal"
            else:
                if (k == 0):
                    label="benign"
                elif(k==1):
                    label="malignant"
                else:
                    label="normal"
            final.append({
                "image": j,
                "predictions": k,
                "class": label,
                "Fact":fa
            })

        display_count = 0
        randomly_pred = final
        random.shuffle(randomly_pred)
        for each in randomly_pred:
            if (display_count < display_max):
                plt.imshow(each['image'])
                plt.suptitle(f"Fact: {each['Fact']}")
                plt.title(f"Predict: {each['class']}")
                if (each['Fact'] == each['class']):
                    plt.title(
                        "SUCCESS", fontsize='small', loc='right', color='blue', fontweight='bold'
                    )
                plt.show()
                display_count += 1







        print("Total Test Accuracy = "+str((totalcorrection/self.testImageCount) * 100))










classification2 = Classification(1)
#1way
# classification2.train_path = r'C:/Users/Administrator/Desktop/train'
#     # set testing paths
# classification2.test_path =  r'C:/Users/Administrator/Desktop/test'
# #2way  benign+mal
# classification2.train_path2=r'C:/Users/Administrator/Desktop/train2'
# classification2.test_path2=r'C:/Users/Administrator/Desktop/test2'
#
#


    # train the model
classification2.trainModel()
    # test model
classification2.testModel(12)  #max30


































































import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from tqdm import tqdm


















x_train = []
y_train = []
x_test = []
y_test = []

# reading and getting features for training data
for image in tqdm(os.listdir(r"C:/Users/Administrator/Desktop/breast_train/")):
    arr = image.split(' ')
    if arr[0] == "normal":
        y_train.append(0)
    if arr[0] == "benign":
        y_train.append(1)
    if arr[0]=="malignant":
        y_train.append(2)

    # reading the image
    img = imread(r"C:/Users/Administrator/Desktop/breast_train/" + image)

    # resizing image
    resized_img = resize(img, (128, 64))

    # creating hog features
    #gray = cv.cvtColor(resized_img, cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 2)
    gray = cv.medianBlur(gray, 5)

    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

    x_train.append(fd)



# reading and getting features for testing data
for image in tqdm(os.listdir(r"C:/Users/Administrator/Desktop/breast_test/")):
    arr = image.split(' ')
    if arr[0] == "normal":
        y_test.append(0)
    if arr[0] == "benign":
        y_test.append(1)
    if arr[0] =="malignant":
        y_test.append(2)

    # reading the image
    img = imread(r"C:/Users/Administrator/Desktop/breast_test/" + image)

    # resizing image
    resized_img = resize(img, (128, 64))

    # creating hog features

    gray = cv.cvtColor(resized_img, cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 2)
    gray = cv.medianBlur(gray, 5)


    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

    x_test.append(fd)

# print('x.shape[0]= ' , x_train.shape[0])
# print('y.shape[0]= ' , y_train.shape[0])

# models that applied
models = ['svc linear one vs one', 'svc linear one vs all', 'rbf kernel svc', 'polynomial kernel svc degree 3']

# training models
C = 0.01
svc_one_vs_one = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
svc_one_vs_all = svm.LinearSVC(C=C).fit(x_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=1, C=C).fit(x_train, y_train)
svc_polynomial = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train)

# calculate accuracy for training
for counter, clf in enumerate((svc_one_vs_one, svc_one_vs_all, rbf_svc, svc_polynomial)):
    predictions = clf.predict(x_train)
    accuracy = np.mean(predictions == y_train)
    print("accuracy of " + models[counter] + "  for training is  " + str(accuracy))

print('*******************************************************************************')
# calculate accuracy for testing
for counter, clf in enumerate((svc_one_vs_one, svc_one_vs_all, rbf_svc, svc_polynomial)):
    predictions = clf.predict(x_test)
    accuracy = np.mean(predictions == y_test)
    print("accuracy of " + models[counter] + "  for testing is  " + str(accuracy))





















import os
import warnings
from random import random

import cv2 as cv

warnings.filterwarnings('ignore')
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from tqdm import tqdm


import cv2 as cv
import numpy as np
from glob import glob
import argparse
#from helpers import *
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
from sklearn.datasets import make_blobs
#import plotly.graph_objs as go
#from plotly.offline import init_notebook_mode, iplot
#from plotly import tools
#import imutils                      #basic image processing functions
from sklearn import *       #mach learn
#from keras import *
from tqdm import tqdm      #create Progress Bars.
import itertools             #iterate over data structures
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,classification_report


x_train = []
y_train = []
x_test = []
y_test = []

# reading and getting features for training data
for image in tqdm(os.listdir(r'C:\Users\Administrator\Desktop\breast_train')):
    arr = image.split(' ')
    if arr[0] == "malignant" or arr[0] == "benign":
        y_train.append(0)
    if arr[0] == "normal":
        y_train.append(1)

    # reading the image
    img = imread(r'C:\Users\Administrator\Desktop\breast_train' + '/' + image)

    # resizing image
    resized_img = resize(img, (128, 64))
    #method
    #resized_img = cv.resize(img,(190, 170),   interpolation=cv.INTER_CUBIC)
    #resized_img = cv.medianBlur(img, 7)  # img


    # creating hog features
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

    x_train.append(fd)


# reading and getting features for testing data
for image in tqdm(os.listdir(r'C:\Users\Administrator\Desktop\breast_test')):
    arr = image.split(' ')
    if arr[0] == "malignant" or arr[0] == "benign":
        y_test.append(0)
    if arr[0] == "normal":
        y_test.append(1)

    # reading the image
    img = imread(r'C:\Users\Administrator\Desktop\breast_test'  + '/' + image)

    # resizing image
    resized_img = resize(img, (128, 64))

    #resized_img = cv.resize(img,(190, 170), interpolation=cv.INTER_CUBIC )
    #resized_img = cv.medianBlur(img, 7)  # img

    # creating hog features
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

    x_test.append(fd)


#print('x.shape[0]= ' , x_train.shape[0])
#print('y.shape[0]= ' , y_train.shape[0])

# models that applied
models = ['svc linear one vs one', 'svc linear one vs all', 'rbf kernel svc', 'polynomial kernel svc degree 3']


#random test
#/////////////////////////////////////////////////////////////////////

def testModel(self, display_max):
    correctClassifications = 0
    self.testImages, self.testImageCount = self.file_helper.preparation(self.x1, self.y1, self.labels1)

    predictions = []
    liofpredict = []
    for word, imlist in self.testImages.items():  # هجيب الصور تيست
        print("processing ", word)
        for im in imlist:
            cl = self.recognize(im)
            liofpredict.append(cl)
            predictions.append({
                'image': im,
                'class': cl,
                'object_name': self.name_dict[str(int(cl[0]))],
                'Fact': word
            })

            if (self.name_dict[str(int(cl[0]))] == word):  # لو صح
                correctClassifications = correctClassifications + 1

    print("Test Accuracy = " + str((correctClassifications / self.testImageCount) * 100))

    # print (predictions) randomly
    display_count = 0
    randomly_pred = predictions
    random.shuffle(randomly_pred)
    for each in randomly_pred:
        if (display_count < display_max):
            plt.imshow(each['image'])
            plt.suptitle(f"Fact: {each['Fact']}")
            plt.title(f"Predict: {each['object_name']}")
            if (each['Fact'] == each['object_name']):
                plt.title(
                    "SUCCESS", fontsize='small', loc='right', color='blue', fontweight='bold'
                )
            plt.show()
            display_count += 1



#/////////////////////////////////////////////////////////////////////
# training models
C = 0.01
svc_one_vs_one = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
svc_one_vs_all = svm.LinearSVC(C=C).fit(x_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=1, C=C).fit(x_train, y_train)
svc_polynomial = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train)

# calculate accuracy for training
for counter, clf in enumerate((svc_one_vs_one, svc_one_vs_all, rbf_svc, svc_polynomial)):
    predictions = clf.predict(x_train)
    accuracy = np.mean(predictions == y_train)
    print("accuracy of " + models[counter] + "  for training is  " + str(accuracy))

print('*******************************************************************************')
# calculate accuracy for testing
for counter, clf in enumerate((svc_one_vs_one, svc_one_vs_all, rbf_svc, svc_polynomial)):
    predictions = clf.predict(x_test)
    accuracy = np.mean(predictions == y_test)
    print("accuracy of " + models[counter] + "  for testing is  " + str(accuracy))



















