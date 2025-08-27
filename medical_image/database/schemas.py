from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from pydantic import Field


# TODO: DB 작업할 데이터를 Validation 하는 용도???
# 현재는 없어도 상관은 없는데... 어떻게 쓰는 건지 확인 필요!

# DiagnosisResult 스키마
class DiagnosisResultBase(BaseModel):
    patient_study_id: int
    probability: Optional[str] = None
    disease_code: str
    diagnosis_date: Optional[datetime] = None
    succeeded: Optional[str] = None
    created_by: Optional[str] = None
    created_date: Optional[datetime] = None
    last_modified_by: Optional[str] = None
    last_modified_date: Optional[datetime] = None
    ai_data: Optional[dict] = None  # ai_data는 JSON 형태로 저장


class DiagnosisResultCreate(DiagnosisResultBase):
    pass


class DiagnosisResultUpdate(DiagnosisResultBase):
    pass


class DiagnosisResult(DiagnosisResultBase):
    id: int
    patient_study: "PatientStudy"  # 관계된 PatientStudy 모델을 포함
    disease: "Disease"  # 관계된 Disease 모델을 포함

    class Config:
        orm_mode = True  # SQLAlchemy 모델을 Pydantic 모델로 변환할 수 있도록 설정


# PatientStudy 스키마
class PatientStudyBase(BaseModel):
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    modality: Optional[str] = None
    image_count: Optional[int] = None
    created_by: Optional[str] = None
    created_date: Optional[datetime] = None
    last_modified_by: Optional[str] = None
    last_modified_date: Optional[datetime] = None
    study_date: Optional[str] = None
    study_time: Optional[str] = None
    api_key: Optional[str] = None
    study_uid: Optional[str] = None
    disease: Optional[str] = None


class PatientStudyCreate(PatientStudyBase):
    pass


class PatientStudyUpdate(PatientStudyBase):
    pass


class PatientStudy(PatientStudyBase):
    id: int
    diagnosis_results: List[DiagnosisResult] = []  # 연관된 진단 결과 목록

    class Config:
        orm_mode = True


# Disease 스키마
class DiseaseBase(BaseModel):
    name: Optional[str] = None
    code: str
    activated: Optional[str] = None
    display_threshold: Optional[float] = None
    inference_threshold: Optional[float] = None
    created_by: Optional[str] = None
    created_date: Optional[datetime] = None
    last_modified_by: Optional[str] = None
    last_modified_date: Optional[datetime] = None


class DiseaseCreate(DiseaseBase):
    pass


class DiseaseUpdate(DiseaseBase):
    pass


class Disease(DiseaseBase):
    id: int
    diagnosis_results: List[DiagnosisResult] = []  # 연관된 진단 결과 목록

    class Config:
        orm_mode = True
