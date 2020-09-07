import os
import sys
import glob
import math
import time
import datetime as dt
import numpy as np
import nibabel as nib
import SimpleITK as sitk
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
from skimage.measure import marching_cubes_lewiner
import model_ver_2_0 as model


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
        # Convert to int16 (from sometimes int16),
        # values should always be low enough (<32k)
        # image = image.astype(np.int16)

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
                    lb = remove_small_objects(lb, min_size=np.max(region_list)*0.25)

                if len(regionprops(lb))!=2 :
                    raise Exception('Could not find proper eyes on the face')

                for region in regionprops(lb):
                    boxes.append(list(region.bbox))

            if ch == 1:  # nose

                result = np.round(results[..., ch])
                lb = label(result, connectivity=1)

                if np.max(lb) > 1:
                    region_list = [region.area for region in regionprops(lb)]       
                    lb = remove_small_objects(lb, min_size=np.max(region_list)*0.25)

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
            ones = self.box_blur(ones, boxes[i], 0)
        
        ones = 1-ones
        pred = pred*ones
        
        cmap = colors.ListedColormap(['None', 'red', 'purple', 'blue', 'yellow', 'green']) #rigt eye 1, left eye 2, nose 3, right ear 4, left ear5
        bounds=[0,1,2,3,4,5,6]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        if axial_plane == 0:

            plt.figure(figsize=(15,10))
            
            plt.subplot(1,3,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)
            
            plt.subplot(1,3,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(1,3,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)
            
        
        elif axial_plane == 1:

            plt.figure(figsize=(15,10))
            
            plt.subplot(1,3,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)
            
            plt.subplot(1,3,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(1,3,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)
            
            
        elif axial_plane == 2:

            plt.figure(figsize=(15,10))
            
            plt.subplot(1,3,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)
            
            plt.subplot(1,3,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)
            plt.subplot(1,3,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)
    
        pic_name = os.path.join(save_path,'label_{}.png'.format(os.path.basename(file_name)))
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
            ones = self.box_blur(ones,boxes[i], 1)
        
        ones = 1-ones
        pred = pred*ones.T
        
        image =image.T
        
        cmap = colors.ListedColormap(['None', 'red', 'purple', 'blue', 'yellow', 'green']) #rigt eye 1, left eye 2, nose 3, right ear 4, left ear5
        bounds=[0,1,2,3,4,5,6]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        if axial_plane == 0:

            plt.figure(figsize=(15,10))
            
            plt.subplot(1,3,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)
            
            plt.subplot(1,3,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(1,3,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)
            
        
        elif axial_plane == 1:

            plt.figure(figsize=(15,10))
            
            plt.subplot(1,3,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)
            
            plt.subplot(1,3,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(1,3,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)
            
            
        elif axial_plane == 2:

            plt.figure(figsize=(15,10))
            
            plt.subplot(1,3,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)
            
            plt.subplot(1,3,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)
            plt.subplot(1,3,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)
    
        pic_name = os.path.join(path,'label_{}.png'.format(os.path.basename(file_name)))
        plt.savefig(pic_name, bbox_inches='tight')
        plt.close('all')

        

    # wipe eyes or nose
    def box_blur(self, im_array, box, option, wth=1):
        # increase or decrease the size of the box by 'wth' times
        if wth != 1:
            for c in range(3):
                mean_ = (box[c]+box[c+3])/2
                box[c] = int(np.round(mean_-wth*(mean_-box[c])))
                box[c+3] = int(np.round(wth*(box[c+3]-mean_)+mean_))
                if box[c] < 0:
                    box[c] = 0
                if box[c+3] > im_array.shape[2-c]:
                    # order : im_array-> x,y,z / box-> z,y,x
                    box[c+3] = im_array.shape[2-c]

        # voxel coordinates must be 'int'
        # if it is dicom files
        if option == 0:
            box_x1 = box[0]
            box_y1 = box[1]
            box_z1 = box[2]
            box_x2 = box[3]
            box_y2 = box[4]
            box_z2 = box[5]
        # or it is nfty files
        else:
            box_z1 = box[0]
            box_y1 = box[1]
            box_x1 = box[2]
            box_z2 = box[3]
            box_y2 = box[4]
            box_x2 = box[5]

        # wipe nose
        blurr_array = 0
        im_array[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2] = blurr_array

        return im_array

    # wipe eyes
    def surface_blur(self, im_array, edge_img, box, wth, dep, option):
        # increase or decrease the size of the box by 'wth' times
        if wth != 1:
            for c in range(3):
                mean_ = (box[c]+box[c+3])/2
                box[c] = int(np.round(mean_-wth*(mean_-box[c])))
                box[c+3] = int(np.round(wth*(box[c+3]-mean_)+mean_))
                if box[c] < 0:
                    box[c] = 0
                if box[c+3] > im_array.shape[2-c]:
                    # order : im_array-> x,y,z / box-> z,y,x
                    box[c+3] = im_array.shape[2-c]

        # voxel coordinates must be 'int'
        # if it is dicom files
        if option == 0:
            box_x1 = box[0]
            box_y1 = box[1]
            box_z1 = box[2]
            box_x2 = box[3]
            box_y2 = box[4]
            box_z2 = box[5]

            mini_array = im_array[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2]
            mini_edge = edge_img[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2]
            processing_area = np.zeros_like(mini_array)

            # blur eye
            where_true = np.where(mini_edge == True)

            for i in range(len(where_true[0])):
                x = where_true[0][i]
                y = where_true[1][i]
                z = where_true[2][i]
                processing_area[x-dep:x+dep,y-dep:y+dep,z-dep:z+dep] = 1
            
            threshold = np.max(ndimage.gaussian_filter(mini_array[processing_area==1],sigma=3))
            mini_array[processing_area==1] = threshold
        # or it is nifti files
        else:
            box_z1 = box[0]
            box_y1 = box[1]
            box_x1 = box[2]
            box_z2 = box[3]
            box_y2 = box[4]
            box_x2 = box[5]

            mini_array = im_array[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2]
            mini_edge = edge_img[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2]
            processing_area = np.zeros_like(mini_array)

            # blur eye
            where_true = np.where(mini_edge == True)

            for i in range(len(where_true[0])):
                x = where_true[0][i]
                y = where_true[1][i]
                z = where_true[2][i]
                processing_area[x-dep:x+dep, y-dep:y+dep, z-dep:z+dep] = 1

            threshold = np.max(ndimage.gaussian_filter(
                mini_array[processing_area == 1], sigma=3))
            mini_array[processing_area == 1] = threshold

        im_array[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2] = mini_array
        return im_array

    # convert image 2D to 3D shape
    def outer_contour_3D(self, image, zoom=1):
        # sort in standard size
        resize_factor = (128/image.shape[0],
                         128/image.shape[1], 128/image.shape[2])
        ima = ndimage.zoom(image, resize_factor, order=0,
                           mode='constant', cval=0.0)

        # make binary cast
        thresh = threshold_triangle(ima)
        imageg = ndimage.median_filter(ima, size=3)
        binary_image = imageg > thresh
        for s in range(ima.shape[0]):
            binary_image[s, :, :] = ndimage.morphology.binary_fill_holes(
                binary_image[s, :, :])
        for s in range(ima.shape[1]):
            binary_image[:, s, :] = ndimage.morphology.binary_fill_holes(
                binary_image[:, s, :])
        for s in range(ima.shape[2]):
            binary_image[:, :, s] = ndimage.morphology.binary_fill_holes(
                binary_image[:, :, s])

        # draw outer contour
        verts, faces, norm, val = marching_cubes_lewiner(binary_image, 0)
        vint = np.round(verts).astype('int')
        contour = np.zeros_like(binary_image)
        for s in vint:
            contour[s[0], s[1], s[2]] = 1

        # shrink contour image cuz of the gaussian_filter we used earlier.
        if zoom != 1:
            c_shape = contour.shape
            zoom_ = ndimage.zoom(contour, zoom, order=0,
                                 mode='constant', cval=0.0)
            zoom_shape = zoom_.shape
            npad = ((int(np.ceil((c_shape[0]-zoom_shape[0])/2)), int((c_shape[0]-zoom_shape[0])/2)),
                    (int(np.ceil((c_shape[1]-zoom_shape[1])/2)),
                     int((c_shape[1]-zoom_shape[1])/2)),
                    (int(np.ceil((c_shape[2]-zoom_shape[2])/2)), int((c_shape[2]-zoom_shape[2])/2)))

            contour_3D = np.pad(zoom_, npad, 'constant', constant_values=(0))
        elif zoom == 1:
            contour_3D = contour

        # Revert to original size
        get_back = (image.shape[0]/128, image.shape[1]/128, image.shape[2]/128)
        contour_3D = ndimage.zoom(
            contour_3D, get_back, order=0, mode='constant', cval=0.0)

        return contour_3D

    # where_do_you_want_to_blur? ex) where = (1,1,1) -> blur(eyes, nose, ears)
    def Deidentification_image_dcm(self, where, dicom_path, dest_path, verif_path, prefix, Model=model):
        '''
        where : list or tuple. Each position stands for eyes nose ears (eyes, nose, ears) 
                If the corresponding position is 1, de-identification process.

        dicom_path : Test set(labled or unlabled) data path. 
        model : Predictive model to be applied.
        '''
        try:
            config = dict()
            config["resizing"] = True
            config["input_shape"] = (128, 128, 128, 1)
            config["num_multilabel"] = 4

            prefix += "_{}"

            list_test_image = glob.glob(dicom_path + '/*.dcm')
            slices = self.load_scan(list_test_image)
            array_img = self.get_pixels(slices)
            original_shape = array_img.shape
            d_type = array_img.dtype
            thresh = threshold_triangle(array_img)

            # Header de identification
            self.header_deidentification(slices, check=False)

            # Make the superior coordinates the direction of increasing.

            X = slices[0][0x00200037].value[0:3]
            X = [float(i) for i in X]

            Y = slices[0][0x00200037].value[3:6]
            Y = [float(i) for i in Y]

            superior = [X[2],  Y[2],  np.cross(X,Y)[2]]
            arg = np.argmax(np.abs(superior))

            if superior[arg] < 0:
                image = self.flip_axis(array_img, (2 - arg))
            else:
                image = array_img

            # load prediction label
            image = model.resize(image)
            image = image.reshape(1, 128, 128, 128, 1)
            results = model.model.predict(image)

            if superior[arg] < 0:
                # +1 : index 0 is batch size
                results = self.flip_axis(results, (3 - arg))

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

            # view label with .png
            if not os.path.isdir(verif_path):
                os.makedirs(verif_path)
            fileName = os.path.basename(dest_path)
            self.dicom_view_label(array_img, results, boxes, (2-arg), verif_path, fileName)

            # blur parts of face
            if where[1]:  # nose
                box = boxes[2]
                array_img = self.box_blur(array_img, box, 0, wth=1.33)

            # make outer contour for mini array.
            edge_img = self.outer_contour_3D(array_img, zoom=1)

            if where[0]:  # eyes

                box = boxes[0]  # eye
                array_img = self.surface_blur(
                    array_img, edge_img, box, wth=1.5, dep=3, option=0)

                box = boxes[1]  # eye
                array_img = self.surface_blur(
                    array_img, edge_img, box, wth=1.5, dep=3, option=0)

            if where[2]:  # ears
                '''
                In order not to see the outline of the ear due to external noise,
                fill the area of the ear with similar noise
                '''
                ear_results = results[...,3]
                border = self.box_blur(np.ones(array_img.shape), boxes[3], 0) #'box_blur' function is based on array_img.shape (nibabel liabrary)
                border = self.box_blur(border, boxes[4], 0)
                border = 1-border
                ear_results = border*ear_results

                noise = np.random.rand(*original_shape)*thresh*0.8 
                array_img[ear_results == 1] = noise[ear_results == 1] 

            array_img = np.round(array_img)
            array_img = np.array(array_img, dtype=d_type)
            # processed 3D image array

            for i in range(len(slices)):
                # [i, :, : ] the reason is that function np.stack makes new axis as first axis
                slices[i].PixelData = array_img[i, :, :].tostring()

            for i in range(len(slices)):
                instanceNum = pydicom.read_file(
                    list_test_image[i]).InstanceNumber
                # instance Number starts from 1.
                slices[instanceNum-1].save_as(os.path.join(
                    dest_path, prefix.format(os.path.basename(list_test_image[i]))))

            return {"success": True, "msg": ""}
        except Exception as ex:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(ex).__name__, ex)
            return {"success": False, "msg": str(ex)}

    # 5D tensor (batch,img_dep,img_cols,img_rows,img_channel)
    def load_batch(self, x_list, y_list=0, batch_size=1):
        # 확인 후 삭제
        config = dict()  # configuration info
        config["resizing"] = True
        config["img_channel"] = 1
        config["batch_size"] = 1
        config["num_multilabel"] = 4  # the number of label (channel last)

        image = sitk.GetArrayFromImage(
            sitk.ReadImage(x_list)).astype('float32')
        if config["resizing"] == True:
            image = self.resize(image)
            img_shape = image.shape
        else:
            img_shape = image.shape

        image = np.reshape(image, (config["batch_size"], img_shape[0], img_shape[1],
                                   img_shape[2], config["img_channel"]))  # batch, z ,y, x , ch

        label = 0
        if y_list != 0:
            labels = sitk.GetArrayFromImage(
                sitk.ReadImage(y_list)).astype('float32')
            if config["resizing"] == True:
                labels = self.resize(labels)
                lb_shape = labels.shape
            else:
                lb_shape = labels.shape

            onehot = to_categorical(labels)
            label = np.reshape(
                onehot, (config["batch_size"], lb_shape[0], lb_shape[1], lb_shape[2], config["num_multilabel"]))

        return image, label

    def resize(self, data, img_dep=128, img_cols=128, img_rows=128):
        resize_factor = (
            img_dep/data.shape[0], img_cols/data.shape[1], img_rows/data.shape[2])
        data = ndimage.zoom(data, resize_factor, order=0,
                            mode='constant', cval=0.0)
        return data

    # where_do_you_want_to_blur? ex) where = (1,1,1) -> blur(eyes, nose, ears)
    def Deidentification_image_nii(self, where, nfti_path, dest_path, verif_path, prefix, Model=model):
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

            # load prediction label
            image, label = self.load_batch(nfti_path)  # z, y, x
            results = model.model.predict(image)
            results = np.round(results)

            # preprocessing: Size recovery and transform onehot to labels number
            if config["resizing"] == True:
                results = self.onehot2label(results)
                # prediction results (batch size, dep, col ,row, ch) -> (dep, col ,row)
                results = np.reshape(results, config["input_shape"][0:3])
                results = self.resize(results,
                                      img_dep=original_shape[2],
                                      img_cols=original_shape[1],
                                      img_rows=original_shape[0])
                # except 0 label (blanck)
                results = to_categorical(results)

            else:
                results = results[0, ...]

            # search center by clustering
            boxes = self.bounding_box(results[..., 1:])

            if len(boxes) != 5:
                raise Exception("Can not find all the parts of the face")

            # view label with .png
            if not os.path.isdir(verif_path):
                os.makedirs(verif_path)
            fileName = os.path.basename(dest_path)
            self.nifti_view_label(array_img, results, boxes, verif_path, fileName)

            # blur parts of face
            if where[1]:  # nose
                box = boxes[2]
                array_img = self.box_blur(array_img, box, 1, wth=1.33)

            # make outer contour for mini array.
            edge_img = self.outer_contour_3D(array_img, zoom=1)

            if where[0]:  # eyes

                box = boxes[0]  # eye
                array_img = self.surface_blur(
                    array_img, edge_img, box, wth=1.5, dep=3, option=1)

                box = boxes[1]  # eye
                array_img = self.surface_blur(
                    array_img, edge_img, box, wth=1.5, dep=3, option=1)

            if where[2]:  # ears
                '''
                In order not to see the outline of the ear due to external noise,
                fill the area of the ear with similar noise
                '''
                ear_results = results[...,3]
                border = self.box_blur(np.ones(array_img.shape), boxes[3], 1) #'box_blur' function is based on array_img.shape (nibabel liabrary)
                border = self.box_blur(border, boxes[4], 1)
                border = 1-border
                ear_results = border*ear_results.T
                
                noise = np.random.rand(*original_shape)*thresh*0.8 
                array_img[ear_results == 1] = noise[ear_results == 1]

            array_img = np.round(array_img)
            array_img = np.array(array_img, dtype='int32')

            nib.save(nib.Nifti1Image(array_img, raw_img.affine, raw_img.header),
                     os.path.join(os.path.dirname(dest_path), prefix.format(os.path.basename(dest_path))))

            return {"success": True, "msg": ""}
        except Exception as ex:
            print(ex)
            return {"success": False, "msg": str(ex)}
