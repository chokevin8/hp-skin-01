# MedicalImage
----
#### 24.08.19. 
- 24.08.14에 업데이트한 normalize 코드를 그대로 사용하면, 영상을 binary 하게 만들어버리는 문제가 발생
- arr의 dtype이 float이 아닌 uint 이기 때문에 발생한 문제임
- 따라서 arr의 dtype을 np.float32로 변환


#### 24.08.14. branch created
- 인하대 DICOM files를 처리할 때, WindowWidth, WindowCenter, VOILUTSequence header가 전부 존재하지 않는 경우 KeyError가 발생하며 처리되지 않는 문제가 발생
- 3개의 header가 없는 DICOM의 경우는 전부 windowing 처리가 된 상태 ([0,255] 또는 [0,4095]의 pixel range를 갖음)
- 따라서 3개의 header가 없는 DICOM은 [0,1] normalize만 적용하도록 코드를 수정함