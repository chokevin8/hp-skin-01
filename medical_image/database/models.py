from sqlalchemy import Column, ForeignKey, Sequence
from sqlalchemy import Integer, String, DateTime, Float
from sqlalchemy.orm import relationship
from sqlalchemy import func

from medical_image.database.config import Base


diagnosis_result_sequence = Sequence('diagnosis_result_seq', start=1, increment=1)
patient_study_sequence = Sequence('patient_study_seq', start=1, increment=1)
disease_sequence = Sequence('disease_seq', start=1, increment=1)


class DiagnosisResult(Base):
    __tablename__ = "diagnosis_result"

    id                  = Column(Integer, diagnosis_result_sequence, primary_key=True, index=True)
    patient_study_id    = Column(Integer, ForeignKey('patient_study.id'), nullable=False)
    probability         = Column(String(20))
    disease_code        = Column(String(50), ForeignKey('disease.code'))
    diagnosis_date      = Column(DateTime, default=func.now())
    succeeded           = Column(String(1))
    created_by          = Column(String(50))
    created_date        = Column(DateTime, default=func.now())
    last_modified_by    = Column(String(50))
    last_modified_date  = Column(DateTime, default=func.now())

    patient_study = relationship("PatientStudy", back_populates="diagnosis_results")
    disease = relationship("Disease", back_populates="diagnosis_results")


class PatientStudy(Base):
    __tablename__ = 'patient_study'

    id                  = Column(Integer, patient_study_sequence, primary_key=True, index=True)
    patient_id          = Column(String(64))
    patient_name        = Column(String(320))
    patient_age         = Column(String(4))
    modality            = Column(String(16))
    image_count         = Column(Integer)
    created_by          = Column(String(50))
    created_date        = Column(DateTime, default=func.now())
    last_modified_by    = Column(String(50))
    last_modified_date  = Column(DateTime, default=func.now())
    study_date          = Column(String(8))
    study_time          = Column(String(14))

    # 추가
    api_key             = Column(String(32))
    study_uid           = Column(String(64))
    
    diagnosis_results = relationship("DiagnosisResult", back_populates="patient_study")


class Disease(Base):
    __tablename__ = "disease"

    id                  = Column(Integer, disease_sequence, primary_key=True, index=True)
    name                = Column(String(50))
    code                = Column(String(50), nullable=False, unique=True)
    activated           = Column(String(1))
    display_threshold   = Column(Float)
    inference_threshold = Column(Float)
    created_by          = Column(String(50))
    created_date        = Column(DateTime, default=func.now())
    last_modified_by    = Column(String(50))
    last_modified_date  = Column(DateTime, default=func.now())

    diagnosis_results = relationship("DiagnosisResult", back_populates="disease")
