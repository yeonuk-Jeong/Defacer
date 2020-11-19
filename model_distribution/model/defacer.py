import tensorflow as tf
import os
import sys
import glob
import math
import time
import datetime as dt
import numpy as np
import nibabel as nib

import random
import pydicom
import scipy.ndimage
from scipy import ndimage
from skimage import morphology
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
from matplotlib import colors
from keras.utils import to_categorical
from skimage.filters import threshold_triangle

import model.model_ver_contour as model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


graph = tf.get_default_graph()


class Defacer(object):

    # onehot results -> argmax
    def onehot2label(self, onehot_array):
        onehot_array = np.argmax(onehot_array, axis=-1)
        label = onehot_array[..., np.newaxis]

        return label

    # Make the superior coordinates the direction of increasing.
    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    # Loop over the image files and store everything into a list.
    def load_scan(self, list_test_image):
        slices = [pydicom.read_file(s)
                  for s in list_test_image if s.endswith(".dcm")]
        slices.sort(key=lambda x: int(x.InstanceNumber))  # stack

        return slices

    # Merge dicom image 2D to 3D
    def get_pixels(self, scans):
        # default stack axis = 0 // pixel_array function import [y, x], 3D array becomes [z y x]
        image = np.stack([s.pixel_array for s in scans])
        

        return image

    # Delete dicom's header info
    def header_deidentification(self, scans, check=True):
        de_code_list = [0x00080012,  # Instance Creation Date
                        0x00080013,  # Instance Creation Time
                        0x00080020,  # Study Date
                        0x00080021,  # Series Date
                        0x00080022,  # Acquisition Date
                        0x00080023,  # Image Date, Content Date
                        0x00080030,  # Study Time
                        0x00080031,  # Series Time
                        0x00080032,  # Acquisition Time
                        0x00080033,  # Image Time, Content Time
                        0x00080050,  # Accession Number
                        0x00080080,  # Institution name
                        0x00080081,  # Institution Address
                        0x00080090,  # Referring Physician's name
                        0x00081010,  # Station name
                        0x00081040,  # Institutional Department name
                        0x00081070,  # Operator's Name
                        0x00100010,  # Patient's name
                        0x00100020,  # Patient's ID
                        0x00100030,  # Patient's Birth Date
                        0x00100040,  # Patient's Sex
                        0x00101010,  # Patient's Age
                        0x00204000]  # Image Comments

        for s in scans:
            for code in de_code_list:
                # If present, replace with spaces
                try:
                    s[code].value = ''
                except:
                    pass

        if check == True:  # check option
            # s[0x00200013].value: Instance Number
            print('dicom Instance Number:', scans[0][0x00200013].value, '\n')
            for code in de_code_list:
                try:
                    print('DE-IDENTIFIED : ', s[code])
                except:
                    pass

    # Find eyes and nods

    def bounding_box(self, results):
        boxes = list()
        for ch in range(results.shape[-1]):  # except 0 label (blanck)
            if ch == 0 or ch == 2:  # eyes, ears

                result = np.round(results[..., ch])
                lb = label(result, connectivity=1)

                if np.max(lb) > 2:
                    region_list = [region.area for region in regionprops(lb)]       
                    lb = remove_small_objects(lb, min_size=np.max(region_list)*0.3)

                if len(regionprops(lb))!=2 :
                    raise Exception('Could not find proper eyes on the face')

                for region in regionprops(lb):
                    boxes.append(list(region.bbox))

            if ch == 1 or ch ==3 : # nose, mouth

                result = np.round(results[..., ch])
                lb = label(result, connectivity=1)

                if np.max(lb) > 1:
                    region_list = [region.area for region in regionprops(lb)]       
                    lb = remove_small_objects(lb, min_size=np.max(region_list)*0.3)

                if len(regionprops(lb))!=1 :
                    raise Exception('Could not find proper nose on the face')

                for region in regionprops(lb):
                    boxes.append(list(region.bbox))

        return boxes
    

    def dicom_view_label (self, image, labels, boxes, axial_plane, save_path, file_name):
        boxes =np.array(boxes)
        centers = (boxes[:,0:3]+boxes[:,3:6])/2 #centers of nose, right ear, left ear
        
        pred = np.argmax(labels,axis=-1)
        
        ones = np.ones(image.shape)
        for i in range(len(boxes)):
            ones = self.box_blur(ones, boxes[i])
        
        ones = 1-ones
        pred = pred*ones
        
        cmap = colors.ListedColormap(['None', 'red', 'purple', 'blue', 'yellow', 'green']) #rigt eye 1, left eye 2, nose 3, right ear 4, left ear5
        bounds=[0,1,2,3,4,5,6]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        if axial_plane == 0:

            plt.figure(figsize=(15,10))
            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int(centers[5][axial_plane])
            plt.title('predicted mouth: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

        elif axial_plane == 1:

            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int((centers[5][axial_plane]))
            plt.title('predicted mouth: axial = {}' .format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)


        elif axial_plane == 2:

            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int(centers[5][axial_plane])
            plt.title('predicted mouth: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)
    
        pic_name = os.path.join(save_path,'label_{}.png'.format(os.path.getctime(file_name)))
        plt.savefig(pic_name, bbox_inches='tight')
        plt.close('all')

    # take a pic for users to check areas that this tool has found
    def nifti_view_label (self, image,labels,boxes,path,file_name): 
        pred = np.argmax(labels,axis=-1)
        
        boxes =np.array(boxes)
        centers = (boxes[:,0:3]+boxes[:,3:6])/2 #centers of nose, right ear, left ear
        axial_plane = np.argmin(np.var(centers[2:5],axis=0)) # 코와 귀 2개 좌표만 뽑아서 가장 분산이 작은 축을 구함 = axial 축일것이라 예상됨.
        
        ones = np.ones(image.shape)
        for i in range(len(boxes)):
            ones = self.box_blur(ones,boxes[i])
        
        ones = 1-ones
        pred = pred*ones
        
        
        
        cmap = colors.ListedColormap(['None', 'red', 'purple', 'blue', 'yellow', 'green']) #rigt eye 1, left eye 2, nose 3, right ear 4, left ear5
        bounds=[0,1,2,3,4,5,6]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        if axial_plane == 0:

            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int(centers[5][axial_plane])
            plt.title('predicted mouth: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

        elif axial_plane == 1:

            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int((centers[5][axial_plane]))
            plt.title('predicted mouth: axial = {}' .format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)
	        
        elif axial_plane == 2:

            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int(centers[5][axial_plane])
            plt.title('predicted mouth: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)
    
        pic_name = os.path.join(path,'label_{}.png'.format(os.path.basename(file_name)))
        plt.savefig(pic_name, bbox_inches='tight')
        plt.close('all')

        

    # wipe eyes or nose
    def box_blur(self, im_array, box, wth=1):
        # increase or decrease the size of the box by 'wth' times
        if wth != 1:
            for c in range(3):
                mean_ = (box[c]+box[c+3])/2
                box[c] = int(np.round(mean_-wth*(mean_-box[c])))
                box[c+3] = int(np.round(wth*(box[c+3]-mean_)+mean_))
                if box[c] < 0:
                    box[c] = 0
                if box[c+3] > im_array.shape[c]:
                    box[c+3] = im_array.shape[c]

        # voxel coordinates must be 'int'
        
        
        box_x1 = box[0]
        box_y1 = box[1]
        box_z1 = box[2]
        box_x2 = box[3]
        box_y2 = box[4]
        box_z2 = box[5]
    


        # wipe nose
        blurr_array = 0
        im_array[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2] = blurr_array

        return im_array


    def label_denoising(self, results) :
        for ch in range(1,results.shape[-1]): # except 0 label (blanck)
            if ch==1 or ch == 3 : # eyes, ears
                result = np.round(results[...,ch]) # only batch size = 1
                lb=label(result,connectivity=1)

                region_list=list()
                for region in regionprops(lb):
                    region_list.append(region.area)

                region_list_sort = sorted(region_list)

                lb=np.array(lb,dtype=bool)
                lb=remove_small_objects(lb, min_size=region_list_sort[-2])
                results[...,ch]=results[...,ch]*lb


            if ch==2 or ch==4: # nose

                result = np.round(results[...,ch])  # only batch size = 1
                lb=label(result,connectivity=1)

                region_list=list()
                for region in regionprops(lb):
                    region_list.append(region.area)

                max_size= np.max(region_list)
                lb=np.array(lb,dtype=bool)
                lb=remove_small_objects(lb, min_size=max_size)
                results[...,ch]=results[...,ch]*lb

        return results
    

   


    # where_do_you_want_to_blur? ex) where = (1,1,1,1) -> blur(eyes, nose, ears, mouth)
    def Deidentification_image_dcm(self, where, dicom_path, dest_path, verif_path, url, prefix, Model=model):
        '''
        where : list or tuple. Each position stands for eyes nose ears mouth 
                If the corresponding position is 1, de-identification process.

        dicom_path : Test set(labled or unlabled) data path. 
        model : Predictive model to be applied.
        '''
        try:
            config = dict()
            config["resizing"] = True
            config["input_shape"] = (128, 128, 128, 1)

            prefix += "_{}"

            list_test_image = glob.glob(dicom_path + '/*.dcm')
            slices = self.load_scan(list_test_image)
            array_img = self.get_pixels(slices)
            original_shape = array_img.shape
            d_type = array_img.dtype
            thresh = threshold_triangle(array_img)


            # Header de identification
            #self.header_deidentification(slices, check=False)

            # Make the superior coordinates the direction of increasing.
            def flip_axis(x, axis):
                x = np.asarray(x).swapaxes(axis, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, axis)
                return x

            X=slices[0][0x00200037].value[0:3]
            X=[float(i) for i in X]

            Y=slices[0][0x00200037].value[3:6]
            Y=[float(i) for i in Y]


            superior = [X[2],  Y[2],  np.cross(X,Y)[2]]
            arg=np.argmax(np.abs(superior))

            if superior[arg] < 0 :
                array_img_re = flip_axis(array_img,(2 - arg))  # becouse of the image coordination is [z, y, x]
            else : array_img_re = array_img

            array_img_re = self.resize(array_img_re)
            array_img_re = np.reshape(array_img_re,(1 ,config["input_shape"][0], config["input_shape"][1], config["input_shape"][2] ,1)) # batch, z ,y, x , ch

            # load prediction label
            with graph.as_default():
            	results = model.model.predict(array_img_re)

            if superior[arg] < 0 :
                results=flip_axis(results,(3 - arg)) # +1 : index 0 is batch size

            

            # preprocessing: Size recovery and transform onehot to labels number
            if config["resizing"] == True:
                results = self.onehot2label(results)
                # prediction results (batch size, dep, col ,row, ch) -> (dep, col ,row)
                results = np.reshape(results, config["input_shape"][0:3])
                results = model.resize(results,
                                       img_dep=original_shape[0],
                                       img_cols=original_shape[1],
                                       img_rows=original_shape[2])
                results = to_categorical(results)

            else:
                results = results[0, ...]  # Only if batch size==1

            # search center by clustering
            boxes = self.bounding_box(results[..., 1:])
            results = self.label_denoising(results)



            # view label with .png
            if not os.path.isdir(verif_path):
                os.makedirs(verif_path)
            fileName = os.path.abspath(dest_path)
            self.dicom_view_label(array_img, results, boxes, (2-arg), verif_path, fileName)

            # blur parts of face
            if where[1]:  # nose
                box = boxes[2]
                array_img = self.box_blur(array_img, box, wth=1.33)


            if where[0]:  # eyes

                eye_results = results[...,1] 
                

                if where[1] == False: # If you want to preserve the nose
                     border = self.box_blur(np.ones(array_img.shape),boxes[2], wth=1.5)
                     eye_results = border*eye_results

                threshold = np.max(ndimage.gaussian_filter(array_img[eye_results==1],sigma=3))
                array_img[eye_results==1] = threshold

            if where[2]:  # ears
                '''
                In order not to see the outline of the ear due to external noise,
                fill the area of the ear with similar noise
                '''
                ear_results = results[...,3]
                # border = self.box_blur(np.ones(array_img.shape), boxes[3]) #'box_blur' function is based on array_img.shape (nibabel liabrary)
                # border = self.box_blur(border, boxes[4])
                # border = 1-border
                # ear_results = border*ear_results

                noise = np.random.rand(*original_shape)*thresh*0.8 
                array_img[ear_results == 1] = noise[ear_results == 1] 

            if where[3] : # mouth

                mouth_results = results[...,4] 
                # border = self.box_blur(np.ones(array_img.shape),boxes[5]) #'box_blur' function is based on array_img.shape (nibabel liabrary)
                # border = 1-border
                if where[1] == False: # If you want to preserve the nose
                     border = self.box_blur(np.ones(array_img.shape),boxes[2], wth=1.5)
                     mouth_results = border*mouth_results

                threshold = np.max(ndimage.gaussian_filter(array_img[mouth_results==1],sigma=3))
                array_img[mouth_results==1] = threshold

            array_img = np.round(array_img)
            array_img = np.array(array_img, dtype=d_type)
            # processed 3D image array

            for i in range(len(slices)):
                # [i, :, : ] the reason is that function np.stack makes new axis as first axis
                slices[i].PixelData = array_img[i, :, :].tostring()

            files = []
            for i in range(len(slices)):
                instanceNum = pydicom.read_file(
                    list_test_image[i]).InstanceNumber
                # instance Number starts from 1.
                slices[instanceNum-1].save_as(os.path.join(dest_path, prefix.format(os.path.basename(list_test_image[i]))))
                files.append(url + prefix.format(os.path.basename(list_test_image[i])))


            return {"success": True, "path": dest_path, "files": files}
        except Exception as ex:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(ex).__name__, ex)
            return {"success": False, "msg": str(ex)}


    
    def resize(self, data, img_dep=128, img_cols=128, img_rows=128):
        resize_factor = (
            img_dep/data.shape[0], img_cols/data.shape[1], img_rows/data.shape[2])
        data = ndimage.zoom(data, resize_factor, order=0,
                            mode='constant', cval=0.0)
        return data


    # where_do_you_want_to_blur? ex) where = (1,1,1,1) -> blur(eyes, nose, ears,mouth)
    def Deidentification_image_nii(self, where, nfti_path, dest_path, verif_path, url, prefix, Model=model):
        '''
        where : list or tuple. Each position stands for eyes nose ears (eyes, nose, ears) 
                If the corresponding position is 1, de-identification process.
        model : Predictive model to be applied.
        '''
        config = dict()  # configuration info
        config["resizing"] = True
        config["input_shape"] = [128, 128, 128, 1]
        prefix += "_{}"

        try:
            # get affine and header of original image file.
            raw_img = nib.load(nfti_path)
            array_img = raw_img.get_fdata()  # image array
            original_shape = array_img.shape  # (x,y,z)
            thresh = threshold_triangle(array_img)

            
            array_img_re = self.resize(array_img)
            

            array_img_re = np.reshape(array_img_re,(1,config["input_shape"][0], config["input_shape"][1], config["input_shape"][2] , 1)) # batch, z ,y, x , ch

            with graph.as_default():
            	results = model.model.predict(array_img_re)
            results = np.round(results)

            # preprocessing: Size recovery and transform onehot to labels number
            if config["resizing"] == True:
                results = self.onehot2label(results)
                # prediction results (batch size, dep, col ,row, ch) -> (dep, col ,row)
                results = np.reshape(results, config["input_shape"][0:3])
                results = self.resize(results,
                                      img_dep=original_shape[0],
                                      img_cols=original_shape[1],
                                      img_rows=original_shape[2])
                # except 0 label (blanck)
                results = to_categorical(results)

            else:
                results = results[0, ...]

            # search center by clustering
            boxes = self.bounding_box(results[..., 1:])
            results = self.label_denoising(results)

            
            # view label with .png
            if not os.path.isdir(verif_path):
                os.makedirs(verif_path)
            fileName = os.path.basename(dest_path)
            self.nifti_view_label(array_img, results, boxes, verif_path, fileName)

            # blur parts of face
            if where[1]:  # nose
                box = boxes[2]
                array_img = self.box_blur(array_img, box, wth=1.33)

            

            if where[0]:  # eyes

                eye_results = results[...,1] 
                border = self.box_blur(np.ones(array_img.shape),boxes[0]) 
                border = self.box_blur(border, boxes[1])
                border = 1-border
                if where[1] == False: # If you want to preserve the nose
                     border = self.box_blur(border,boxes[2], wth=1.5)

                eye_results = border*eye_results

                threshold = np.max(ndimage.gaussian_filter(array_img[eye_results==1],sigma=3))
                array_img[eye_results==1] = threshold

            if where[2]:  # ears
                '''
                In order not to see the outline of the ear due to external noise,
                fill the area of the ear with similar noise
                '''
                ear_results = results[...,3]
                border = self.box_blur(np.ones(array_img.shape), boxes[3]) 
                border = self.box_blur(border, boxes[4])
                border = 1-border
                ear_results = border*ear_results
                
                noise = np.random.rand(*original_shape)*thresh*0.8 
                array_img[ear_results == 1] = noise[ear_results == 1]

            if where[3] : # mouth
            
	            mouth_results = results[...,4] 
	            border = self.box_blur(np.ones(array_img.shape),boxes[5]) 
	            border = 1-border
	            if where[1] == False: # If you want to preserve the nose
	                 border = self.box_blur(border,boxes[2], wth=1.5)
	                    
	            mouth_results = border*mouth_results
	            
	            threshold = np.max(ndimage.gaussian_filter(array_img[mouth_results==1],sigma=3))
	            array_img[mouth_results==1] = threshold
	            

            array_img = np.round(array_img)
            array_img = np.array(array_img, dtype='int32')

            files = []
            nib.save(nib.Nifti1Image(array_img, raw_img.affine, raw_img.header),
                     os.path.join(os.path.dirname(dest_path), prefix.format(os.path.basename(nfti_path))))
            files.append(url + prefix.format(os.path.basename(os.path.basename(nfti_path))))

            return {"success": True, "path": dest_path, "files": files}
        except Exception as ex:
            print(ex)
            return {"success": False, "msg": str(ex)}
