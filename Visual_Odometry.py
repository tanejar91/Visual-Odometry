import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from helper import *

# path to data
root_dir_path = os.getcwd()
rgb_path = os.path.join(root_dir_path, 'data/rgb')
num_images = 52         #total number of images 
rgb_images = []
for i in range(1,num_images+1):
    img = os.path.join(rgb_path,'frame_'+'{0:05}'.format(i)+'.png')
    # storing paths of images in a rgb_images
    rgb_images.append(img)

class VisualOdometrey:
    def __init__(self):
        pass

    ########## Keypoints and descriptors generation ##########
    def generate_features(self,img):
        '''
        Generate keypoints and features for an image

        Arguments:
        img -- image for which keypoints and descriptors will be extracted 
        
        Returns:
        kp -- keypoints(coordiantes) of the important features
        des -- descriptors assocaited with keypoints

        Note:
        Here I have used ORB as feature detector as SURF is not availble for python 3.6 for Windows. If you can use SURF then 
        use it as it gave better results. 

        '''


        # Feature Generation using ORB
        orb = cv2.ORB_create()
        kp = orb.detect(img,None)
        kp, des = orb.compute(img, kp)

        # Feature Generation using ORB
    #     surf = cv2.xfeatures2d.SURF_create(1000)
    #     kp, des = surf.detectAndCompute(img,None)

        return kp,des

    ########## keypoints and descriptors generation for whole data set ##########
    def generate_features_dataset(self,images):
        '''
        Generate keypoints and features for all the images in the dataset

        Arguments:
        images -- list of paths of all the images in the dataset
        
        Returns:
        kp_list -- list of keypoints(coordiantes) for every image
        des_list -- list of descriptors assocaited with keypoints for every image

        '''
        kp_list = []
        des_list = []
        for i in range(len(images)):
            image = cv2.imread(images[i])
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp,des  = self.generate_features(gray_img)
            kp_list.append(kp)
            des_list.append(des)
        return kp_list,des_list


    ########## match features and filter them using Lowe's ratio test ##########
    def match_features(self,des1,des2,dist_threshold=0.9,num_des=500):

        '''
        Match festures between 2 subsequent images and filter matches features uisng Lowe's ratio test

        Arguments:
        des1 -- descriptors associated with image 1
        des2 -- descriptors associated with image 2 
        dist_threshold -- distance threshold for Lowe's ratio test
        num_des -- number of matches to be returned after filtering

        Returns:
        filtered_match - mtaches filtered after Lowe's ratio test

        '''

        # for  features obtained using ORB
        bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
        matches = bf.knnMatch(des1,des2, k=2)
        
        # for  features obtained using SURF
    #     FLANN_INDEX_KDTREE = 0
    #     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #     search_params = dict()      
    #     flann = cv2.FlannBasedMatcher(index_params,search_params)
    #     matches = flann.knnMatch(des1,des2,k=2)
        
        filtered_match = []
        #Lowe's ratio Test
        for m,n in matches:
            if m.distance < dist_threshold*n.distance:
                filtered_match.append(m)
        filtered_match = sorted(filtered_match, key = lambda x:x.distance)

        return filtered_match[:num_des]

    ########## Visualize matches between a pair of subsequent images ##########
    def visualize_mathes(self,i, n,save = True):

        '''
        Displays(and optionally save) the feature matches between 2 consecutive images

        Arguments:
        i -- image index, must of less than equal to total number of images(52) - 1
        n -- number of matches to be displayed
        save -- if image with matches to be saved

        Returns:
        '''
        des1 = des_data[i]
        des2 = des_data[i+1]
        kp1 = kp_data[i]
        kp2 = kp_data[i+1]
        image1 = cv2.imread(rgb_images[i],0)
        image2 = cv2.imread(rgb_images[i+1],0)
        match = self.match_features(des1,des2)
        image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match[:n],None,flags = 2)
        if save:
            print('Saving matches')
            plt.imsave('matched features between image {} and {}'.format(i,i+1),image_matches)
        plt.figure(figsize=(16, 6), dpi=100)
        plt.imshow(image_matches)
        plt.show()

    ########## Camera movement between a pair of subsequent images ##########
    def visualize_camera_movement(self,i,kp_data,pts_show = 100,save = True):

        '''
        Displays camera movemt between 2 subsequent images by dispalying features movement between images

        Arguments:
        image1 -- first image
        image1_points -- keypoints for matched features for first image
        image2 -- second image 
        image2_points -- keypoints for matched features for second image
        save -- if images needs to be saved
        pts_show -- number of points to be displayed 

        Returns:
        '''

        image1 = cv2.imread(rgb_images[i])
        image2 = cv2.imread(rgb_images[i+1])
        kp1,kp2 = kp_data[i], kp_data[i+1]
        image1_points = [kp1[m.queryIdx].pt for m in match_data[i]] 
        image2_points = [kp2[n.trainIdx].pt for n in match_data[i]]
        image1 = image1.copy()
        image2 = image2.copy()
        for i in range(0, pts_show):
            # Coordinates of a point on t frame
            p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
            # Coordinates of the same point on t+1 frame
            p2 = (int(image2_points[i][0]), int(image2_points[i][1]))
            cv2.circle(image1, p1, 5, (0, 0, 255), 2)
            cv2.arrowedLine(image1, p1, p2, (0, 255, 0), 2)
            cv2.circle(image1, p2, 5, (0, 255, 0), 2)
        if save:
            print('Saving camera visualization')
            plt.imsave('Camera movement between images' ,cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        plt.figure(figsize=(16, 6), dpi=100)
        plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        plt.show()
        

    ###### Trajectory Estimation ########
    def estimate_motion_EssentialMat(self, match, kp1, kp2, k):
        """
        Estimate camera motion from a pair of subsequent image frames

        Arguments:
        match -- list of matched features from the pair of images
        kp1 -- list of the keypoints in the first image
        kp2 -- list of the keypoints in the second image
        k -- camera calibration matrix 
        

        Returns:
        rmat -- recovered 3x3 rotation numpy matrix
        tvec -- recovered 3x1 translation numpy vector
        image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                         coordinates of the i-th match in the image coordinate system
        image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                         coordinates of the i-th match in the image coordinate system
                   
        """

        
        ### START CODE HERE ###
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        image1_points = []
        image2_points = []
        
        ### START CODE HERE ###    
        image1_points = np.array([kp1[m.queryIdx].pt for m in match]) 
        image2_points = np.array([kp2[n.trainIdx].pt for n in match])
        E, mask = cv2.findEssentialMat(image1_points,image2_points,k)
    #     E, mask = cv2.findEssentialMat(image1_points,image2_points, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
        points, rmat, tvec, mask = cv2.recoverPose(E, image1_points,image2_points,k)
        
        return rmat, tvec


    def estimate_trajectory(self, estimate_motion_EssentialMat, matches, kp_list, k):
        """
        Estimate complete camera trajectory from subsequent image pairs

        Arguments:
        estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
        matches -- list of matches for each subsequent image pair in the dataset. 
                   Each matches[i] is a list of matched features from images i and i + 1
        des_list -- a list of keypoints for each image in the dataset
        k -- camera calibration matrix 
        
        Optional arguments:
        depth_maps -- a list of depth maps for each frame. This argument is not needed if you use Essential Matrix Decomposition

        Returns:
        trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and   
                      trajectory[:, i] is a 3x1 numpy vector, such as:
                      
                      trajectory[:, i][0] - is X coordinate of the i-th location
                      trajectory[:, i][1] - is Y coordinate of the i-th location
                      trajectory[:, i][2] - is Z coordinate of the i-th location
                      
                      * Consider that the origin of your trajectory cordinate system is located at the camera position 
                      when the first image (the one with index 0) was taken. The first camera location (index = 0) is geven 
                      at the initialization of this function

        """
        trajectory = np.zeros((3, 1))
        C = np.eye(4)
        lrow = np.array([[0,0,0,1]])

        for i in range(len(matches)):
    #         print(i)
            match = matches[i]
            kp1,kp2 = kp_list[i], kp_list[i+1]
            
            r,t = estimate_motion_EssentialMat(match,kp1,kp2,k)
            T = np.concatenate((r,t),axis=1)        
            T = np.concatenate((T,lrow),axis=0)
            C = np.matmul(C,np.linalg.inv(T))
            trajectory = np.concatenate((trajectory,C[:3,3].reshape(-1,1)),axis=1)
        
        return trajectory


i = 50
n = 30 #num of matches to be displayed

if __name__ == '__main__':

    odom  = VisualOdometrey()
    kp_data,des_data = odom.generate_features_dataset(rgb_images)

    odom.visualize_mathes(i,n)
    # # features match for the whole dataset
    match_data = []
    for i in  range(num_images-1):
        match = odom.match_features(des_data[i],des_data[i+1])
        match_data.append(match)

    odom.visualize_camera_movement(i,kp_data)
    trajectory = odom.estimate_trajectory(odom.estimate_motion_EssentialMat,match_data,kp_data,k)
    # print(trajectory)
    visualize_trajectory(trajectory)