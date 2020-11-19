# model_distribution

defacer_contour가 주 실행 코드이고 모델을 model_ver_contour에서 불러옵니다.  
  
model_contour4.h5 파일이 딥러닝 모델 weight가 들어간 파일이고 이것을 업데이트 하면 모델이 업데이트 됩니다.  
  
defacer_contour.py에서 model_ver_contour.py호출  /  model_ver_contour.py에서 model_contour4.h5 호출
  
  
두경부 T1 MRI 영상을 인풋으로 하여 얼굴을 인식하고 삭제하는 프로그램.    https://github.com/yeonuk-Jeong/Defacer 의 version2 그림 참조.  

  
니프티 파일 (.nii.gz or .nii) 의 경우 Deidentification_image_nii 함수를 호풀해서 처리하고,     
다이콤 (.dcm) 의 경우 Deidentification_image_dcm 함수를 호출해서 처리함.   
where =(1,1,1,1) # 눈 코 귀 입 모두 지움 (0, 0, 0, 0) 은 모두 안지움을 의미합니다. 기본으로 (1,1,1,1)으로 세팅해 주시면 됩니다.     
dest_path, verif_path, prefix는 결과파일 저장위치, 검증파일 저장위치, 결과파일에 붙일 접두어를 각각 의미함.  

 dicom_path ,nfti_path 는  인풋파일 의미 

   

필요라이브러리  

tensorflow==1.14.0  
Keras=2.2.4 

scikit-image==0.15.0  
scikit-learn==0.20.3  
scipy==1.4.1  
numpy==1.16.2  
pydicom==1.2.2  
nibabel==2.4.0  
matplotlib==3.0.3  

## 함수설명  
  
다이콤(.dcm) 보다 니프티(.nii.gz) 포맷이 더 성능이 좋고 편리해서 학습이나 테스트 시연은 니프티 형식의 파일로 하려고합니다.   
두경부 MRI 니프티 파일이 인풋으로 들어가면 영상처리되어 익명화된 두경부 MRI 니프티 파일이 아웃풋으로 나오는 프로그램입니다.  
  
  
Deidentification_image_nii(self, where, nfti_path, dest_path, verif_path, url, prefix, Model=model):  
에서 
where =(1,1,1,1) # 눈 코 귀 입 모두 지움 (0, 0, 0, 0) 은 모두 안지움을 의미합니다. 기본으로 (1,1,1,1)으로 세팅해 주시면 됩니다.       
dest_path, verif_path, prefix는 결과파일 저장위치, 검증파일 저장위치, 결과파일에 붙일 접두어를 각각 의미함.    
nfti_path는 인풋 파일로 들어갈 파일의 경로입니다.  
url은 다운로드 경로를 넘겨줄 주소 입니다.
  
  
아웃풋으로는 이 함수 제일 아래에 nib.save 함수가 영상처리된 니프티 파일을 저장하고, 중간에 nifti_view_label 함수가  verif_path에 검증 이미지를 저장합니다.  
함수의 리턴값은 return {"success": True, "path": dest_path, "files": files}로 성공여부와 url+파일경로를 리턴합니다.  



  

  