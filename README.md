Asan medical center medical image defacer(de-identification)
# De-facer: De-identifier reconstructable facial information in Medical image (CT, MRI) 

### To protect individually identifiable health information through permitting only certain uses, metadata in image headers and reconstructable personal data like 3D facial image must be de-identified.   

This source code contains an algorithm for detecting and removing identifiable facial features (eyes, nose, and ears).  
3D Unet-based neural network were used to detect eyes, nose and ears in the input medical image
Creating bounding box at the largest segmented points that location of each predicted feature and __only removing suface facial features to avoid impinging on region of interest__ such as brain.     
this will be help with the secondary use of medical data by obscuring it locally to minimize information loss.  
The Nifti version includes both training and application, and the Dicom version uses a model that has already been trained to create a de-identified file.  
  
Users could select the combination of facial parts (eyes, nose and ears) to de-identify and this function would be helpful by minimizing information loss.    
  &nbsp;
- Selective de-identification of facial features (eyes, nose, and ears)
- Removing metadata with privacy in dicom header 
- Supporting multi medical image types (nifti, dicom)  
  
## prerequisites
- python (3.x)
- tensorflow
- Keras (Using tensorflow backend)

## Comping up 
- integrating file transfer function with de-id 
- Web-based service
- Supporting CT images

  
# Model training 
## Data
AIBL neuroimage data Nifty file.  
240 * 256 * 160 image T1w MPRage  
training set : 247, 28 validation set : 28, test set : 28

## Labeling
![labeling_sample](https://user-images.githubusercontent.com/49013508/63914266-a03b5600-ca6d-11e9-9331-55ac64f62b84.png)
  
labeling sample. the spherical point at the facial feature.  

## Data augmentation
Rotataion, shift, filp, shear on each axis randomly.  
Randomly zoom in or zoom out in the range of [0.9, 1.10] 
I modified a module keras.preprocessing.image and use ImageDataGenerator function to do data augmentation.  
  
See keras_image_preprocessing1.py for detail.
                                        
## Model structure
![model_pic](https://user-images.githubusercontent.com/49013508/63914225-784bf280-ca6d-11e9-89c7-6d63b63db40d.png)  
  
![attention_gate](https://user-images.githubusercontent.com/49013508/63914263-9ca7cf00-ca6d-11e9-9f53-50d2e769265a.jpg)
  
I modify attention gated U-net[1] as introduce Convolution block.  
Convolution block consists of Convolution(3x3) - Instance Normalization (because batch size is 1) - ReLU - Convolution(3x3) with residual SE net - Instance Normalization - ReLU  Segmentation map is made through 1x1 convoultion with number of channels equal to the number of channels of one hot label data. and activate sofmax.  
  
[1] Schlemper, J., Oktay, O., Schaap, M., Heinrich, M., Kainz, B., Glocker, B., & Rueckert, D. (2019). Attention gated networks: Learning to leverage salient regions in medical images. Medical image analysis, 53, 197-207.
&nbsp;
## Results
![collage1](https://user-images.githubusercontent.com/49013508/71356302-23f0f380-25c5-11ea-90ba-63f3fcad6d49.png)  
![collage2](https://user-images.githubusercontent.com/49013508/71356330-3f5bfe80-25c5-11ea-9229-a0fe982fd8d8.png)

Removing facial feature sample -ITK sanp viewer 
  
![remove_sample_MRIcron_rendering](https://user-images.githubusercontent.com/49013508/63914279-a6313700-ca6d-11e9-862c-3b1d3c96ad4c.png)  
  
Removing facial feature sample -MRI cron rendering 
