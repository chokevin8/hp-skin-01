import daiquiri
from datetime import datetime

from medical_image.database.models import PatientStudy, DiagnosisResult, Disease
from medical_image.database.config import SessionLocal


logger = daiquiri.getLogger("database")


def make_row_with_api(api_key: str):
    new_patient_study = PatientStudy(
        patient_id="",
        patient_name="",
        patient_age="",
        modality="",
        image_count=0,
        study_date="",
        study_time="",
        created_by="anonymousUser",
        created_date=datetime.now(),
        last_modified_by="anonymousUser",
        last_modified_date=datetime.now(),
        api_key=api_key,
        study_uid=""
    )
    try:
        with SessionLocal() as _db:
            _db.add(new_patient_study)
            _db.commit()
            _db.refresh(new_patient_study)

        logger.info(f"Successfully inserted PatientStudy: ID={new_patient_study.id}, Patient ID={new_patient_study.patient_id}, Name={new_patient_study.patient_name}")
    except Exception as exc:
        logger.error('Unknown error while try to insert patient_study')
        logger.exception(exc)
        return None

    return new_patient_study.id


def update_patient_study(patient_study_id: int, patient_id: str, patient_name: str, patient_age: str, modality: str, image_count: int, study_date: str, study_time: str, api_key: str, study_uid: str, created_by: str = 'anonymousUser', last_modified_by: str = 'anonymousUser'):
    try:
        with SessionLocal() as _db:
            patient_study_row = _db.query(PatientStudy).filter(PatientStudy.id == patient_study_id).first()
            if patient_study_row:
                patient_study_row.patient_id = str(patient_id)
                patient_study_row.patient_name = str(patient_name)
                patient_study_row.patient_age = str(patient_age)
                patient_study_row.modality = str(modality)
                patient_study_row.image_count = int(image_count)
                patient_study_row.study_date = str(study_date)
                patient_study_row.study_time = str(study_time)
                patient_study_row.api_key = str(api_key)
                patient_study_row.study_uid = str(study_uid)
                patient_study_row.last_modified_by = str(last_modified_by)
                patient_study_row.last_modified_date = datetime.now()
                _db.commit() 

        logger.info(f"Successfully updated PatientStudy: ID={patient_study_id}, Patient ID={patient_id}, Name={patient_name}")
    except Exception as exc:
        logger.error('Unknown error while try to update patient_study')
        logger.exception(exc)
        return None
    
    return patient_study_row


def insert_patient_study(patient_id: str, patient_name: str, patient_age: str, modality: str, image_count: int, study_date: str, study_time: str, api_key: str, study_uid: str, created_by: str = 'anonymousUser', last_modified_by: str = 'anonymousUser'):
    """
    새로운 PatientStudy 데이터를 삽입하는 함수

    :param patient_id: 환자 ID
    :param patient_name: 환자 이름
    :param patient_age: 환자 나이
    :param modality: 검사 방식
    :param image_count: 이미지 개수
    :param study_date: 검사 날짜 (형식: YYYYMMDD)
    :param study_time: 검사 시간 (형식: HHMMSS)
    :param created_by: 생성자
    :param last_modified_by: 마지막 수정자
    :return: 생성된 PatientStudy 객체
    """
    new_patient_study = PatientStudy(
        patient_id=patient_id,
        patient_name=patient_name,
        patient_age=patient_age,
        modality=modality,
        image_count=image_count,
        study_date=study_date,
        study_time=study_time,
        created_by=created_by,
        created_date=datetime.now(),
        last_modified_by=last_modified_by,
        last_modified_date=datetime.now(),
        api_key=api_key,
        study_uid=study_uid
    )
    try:
        with SessionLocal() as _db:
            _db.add(new_patient_study)
            _db.commit()
            _db.refresh(new_patient_study)

        logger.info(f"Successfully inserted PatientStudy: ID={new_patient_study.id}, Patient ID={new_patient_study.patient_id}, Name={new_patient_study.patient_name}")
    except Exception as exc:
        logger.error('Unknown error while try to insert patient_study')
        logger.exception(exc)
        return None

    return new_patient_study.id


def insert_diagnosis_result(patient_study_id: int, probability: str, disease_code: str = None, succeeded: str = 'N', created_by: str = 'anonymousUser', last_modified_by: str = 'anonymousUser'):
    """
    새로운 DiagnosisResult 데이터를 삽입하는 함수

    :param patient_study_id: 환자 연구 ID
    :param probability: 확률 값
    :param disease_code: 질병 코드
    :param succeeded: 완료 상태 (Y/N)
    :param created_by: 생성자
    :param last_modified_by: 마지막 수정자
    :param ai_data: AI 분석 결과 (JSON 형태)
    :return: 생성된 DiagnosisResult 객체
    """
    new_diagnosis_result = DiagnosisResult(
        patient_study_id=patient_study_id,
        probability=probability,
        disease_code=disease_code,
        succeeded=succeeded,
        created_by=created_by,
        created_date=datetime.now(),
        last_modified_by=last_modified_by,
        last_modified_date=datetime.now()
    )
    try:
        with SessionLocal() as _db:
            patient_study = _db.query(PatientStudy).filter(PatientStudy.id == patient_study_id).first()
            if not patient_study:
                logger.error(f"PatientStudy with ID '{patient_study_id}' not found")
                raise ValueError(f"PatientStudy with ID '{patient_study_id}' not found")

            _db.add(new_diagnosis_result)
            _db.commit()
            _db.refresh(new_diagnosis_result)

        logger.info(f"Successfully inserted DiagnosisResult: ID={new_diagnosis_result.id}, Patient Study ID={new_diagnosis_result.patient_study_id}, Disease Code={new_diagnosis_result.disease_code}, Probability={new_diagnosis_result.probability}")

    except Exception as exc:
        logger.error('Unknown error while try to insert diagnosis_result')
        logger.exception(exc)

    return new_diagnosis_result


def get_inference_threshold_by_disease_code(disease_code: str) -> float:
    """
    주어진 질병 코드에 해당하는 disease의 inference_threshold 값을 반환하는 함수.

    :param disease_code: 질병 코드 (code)
    :return: 해당 질병의 inference_threshold 값 (없으면 None)
    """
    # Session을 열고, Disease 테이블에서 해당 질병 코드에 대한 inference_threshold 값을 조회합니다.
    with SessionLocal() as _db:
        disease = _db.query(Disease).filter(Disease.code == disease_code).first()

        # 결과가 없다면 None을 반환
        if not disease:
            logger.error(f"Disease with code '{disease_code}' not found")
            return None
        
        # inference_threshold 값 반환
        return disease.inference_threshold


def get_disease_code_by_disease_info(info: str) -> float:
    with SessionLocal() as _db:
        # disease = _db.query(Disease).filter(Disease.activated == "Y" , Disease.code == info).first()
        disease = _db.query(Disease).filter(Disease.code == info).first()

        # 결과가 없다면 None을 반환
        if not disease:
            logger.error(f"Disease with code '{info}' not found")
            return None

        if disease.activated == 'N':
            logger.info(f"Disease with code '{info}' is inactive")
            return None

        # inference_threshold 값 반환
        return disease.code


def get_study_uid_list(api_key: str, patient_id: str) -> list:
    """
    주어진 api_key, patient_id로 patient_study 테이블을 조회하여,
    study_date와 study_time을 합쳐 datetime 객체를 생성한다.
    
    - study_date와 study_time 둘 다 존재하고 비어있지 않다면 datetime 변환
      ㄴ 변환 실패 또는 둘 중 하나라도 존재하지 않거나 빈 문자열이면 created_date를 datetime으로 사용

    Return:
        list of dict: 각 row(=study)의 정보를 담은 딕셔너리들의 리스트
            예: [
                {
                    "datetime": datetime(2025, 1, 21, 13, 20, 59),
                    "study_uid": "1.2.840.113619.2.55.3.2831164357.781.1675968233.467"
                },
                ...
            ]
    """
    with SessionLocal() as _db:
        rows = _db.query(PatientStudy) \
                  .filter(
                        PatientStudy.api_key == api_key,
                        PatientStudy.patient_id == patient_id
                  ) \
                  .order_by(PatientStudy.created_date.asc()) \
                  .all()

        select_list = []
        for row in rows:
            # Only attempt parsing if both study_date and study_time exist and are not empty
            if row.study_date and row.study_time:
                date_str = row.study_date.strip()
                time_str = row.study_time.strip()

                # If neither string is empty, proceed with parsing
                if date_str and time_str:
                    try:
                        yyyy = int(date_str[0:4])
                        mm   = int(date_str[4:6])
                        dd   = int(date_str[6:8])

                        hh = int(time_str[0:2]) if len(time_str) >= 2 else 0
                        mn = int(time_str[2:4]) if len(time_str) >= 4 else 0
                        ss = int(time_str[4:6]) if len(time_str) >= 6 else 0

                        dt = datetime(yyyy, mm, dd, hh, mn, ss)
                    except Exception as exc:
                        logger.warning("Failed to parse study_date/time. Using created_date!")
                        logger.exception(exc)
                        dt = row.created_date
                else:
                    # If either date_str or time_str is empty, use created_date
                    dt = row.created_date
            else:
                # If either is None, use created_date
                dt = row.created_date

            select_list.append({
                "datetime":  dt,
                "study_uid": row.study_uid
            })

        return select_list
