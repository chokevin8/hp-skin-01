import os
import shutil
import cv2
from pathlib import Path
import json
import uuid
import copy
import base64
import pydicom
import daiquiri
import numpy as np
from io import BytesIO
from pydicom.dataset import Dataset
from PIL import ImageFont, Image, ImageDraw
import medical_image
from medical_image.utils import *

op_mode_origin = os.getenv('OP_MODE', 'SECURED-COMPOSE-FILE&DB')
op_mode = op_mode_origin.split('-')[0] or 'SECURED'
output_mode = op_mode_origin.split('-')[2] or 'FILE&DB'

if op_mode == 'SECURED':
    import security_helper.decrypt as Decrypt

if output_mode != 'FILE':
    import medical_image.database.crud as crud

import warnings
warnings.filterwarnings("ignore", message="Invalid value for VR")
warnings.filterwarnings("ignore", message="getsize is deprecated")
warnings.filterwarnings("ignore", message="torch.meshgrid:")


logger = daiquiri.getLogger("ai")


def init_json():
    return {
            "Height": None,
            "Width": None,
            "GraphicAnnotationSequence": [
                # {
                #     "GraphicLayer": None,
                #     "TextObjectSequence": [],
                #     "GraphicObjectSequence": [],
                #     "Probability": None,
                #     "Heatmap": None
                # }
            ],
            "SCArray": None,
            "ReportArray": None,
            "ImageComments": None,
            "Skipping": False
    }


def init_json_3D():
    return {
            "GraphicAnnotationSequence": [
                # {
                #     "GraphicLayer": None,
                #     "Objects": [],
                #     "Probability": None,
                # }
            ],
            "SCArray": None,
            "ReportArray": None,
            "ImageComments": None,
            "Skipping": False
    }


def init_json_patho():                  # TODO: 추후 result_json에 대한 구조는 결정할 예정. 이후 업데이트 예정
    return {
        # "type": "IMG",
        # "scale_ratio": 16,
        # "heat_img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AP5/6KKKKKKKKKKKKKKK/9k=",
        # "detail_report": "download_link",
        # "heatmap": "SUkqABYCAACAP+BACCQWDQeEQmFQuGQ2HQWBP+HxOKRWLQiIwgplNVQtTqcplksq2SFaNRySwVeLxvJdLsyESmCRuOwqZQaMzOUK2TQcAgFAwI/z9AxShQWiRh/0OgRClydVTecgCaSWkw2jwSrgCsgCiUKv0+tU2FUKqzynRKdVGeVuHWCyVyxRW3QezTue3K1VSOK5XNe33O63I/0i407C2Oiw0pFIU1KBwR9Pp+gYDAPDYucUut0I1mtbJZLMulYmvYesUvJ5XL2mCzS/Ne4UHOXGu5mu4PEZmF7eCVPYX/fafNaXeXa57ja7TTWHTa617HZ7vFYjdQd3u87hEIonc0DpwQ+n0fIJBEHoQbO8vh8rn9Xf4LbcuEY3H2j43uF4Ph+umM09rFNnALNr2s4rP8i6EOuhLws2969Nevq/tLBjeuShTnP4pcDvS46Hu+4r4v/C8SRHBLkP1AT6OREzCPdDTUQfBagPsyEVIS/znIdFDiQJHr0uCwEWxo5kPwHFkHxjIyysjHLwPlJiGR0+bmyhF0CKm1YDgOQkSx2n0ZK6HwfFAYZhnCwL/xs/EIujCkFMJMDiOM+ElRqx0bwktkERlEr3UBFcXNLDr8z2q0xSTAkYSvDcIQvAy8SLEMHTrKcq0DAqoI8kDLAGmgVhWCrzPQg9CgAPw/F+apqnWkqaIcm8PTjWlaomqdbVzXSEICAADAAAAQMAAQAAABwAAAABAQMAAQAAABwAAAACAQMAAwAAAKwCAAADAQMAAQAAAAUAAAAGAQMAAQAAAAIAAAARAQQAAQAAAAgAAAAVAQMAAQAAAAMAAAAWAQMAAQAAABwAAAAXAQQAAQAAAA0CAAAcAQMAAQAAAAEAAAA9AQMAAQAAAAIAAABTAQMAAwAAALICAAAAAAAACAAIAAgAAQABAAEA",
        # "opacity": 0.4,
        # "threshold": 0.5,
        # "short_report": {
        #     "Normal_ratio": 100.0,
        #     "Abnormal_ratio": 0.0
        # }
    }

def test_2D(result_json):               # TODO: 사용 여부 확인 필요
    assert result_json["Height"]
    return True


class DicomModule(object):
    def __init__(self, web_json, device=None, gpu_index=None, logger=None):
        self.logger = logger
        # self.output_mode = op_mode_origin.split('-')[2] or 'FILE&DB'
        self.default_graphic_layer_sequence = [
            get_graphic_layer_dataset('LOGO', 0),
            get_graphic_layer_dataset('NO FINDING', 1)
        ]
        self.graphic_layer_sequence = []
        
        try:
            self.read_config(web_json)
            self.device = os.environ['DEVICE'] if device is None else device
            self.gpu_index = os.environ['GPU_IDX'] if gpu_index is None else gpu_index
            self.init_models()

            self.product_name = web_json["product_name"] if web_json["product_name"] else 'TEST'
            self.app_name = web_json["model_name"] if web_json["model_name"] else 'TEST'
            self.app_version = web_json["info"]["udi"]["udi_pi"] if web_json["info"]["udi"]["udi_pi"] else 'TEST'

            self.logo = Image.open(os.path.join(medical_image.__path__[0], 'logo/deepnoid.png'))
            self.colorbar = Image.open(os.path.join(medical_image.__path__[0], 'logo/colorbar.png'))
            self.preferred_character_set = web_json.get('advanced', {}).get('preferred_character_set', '')
        except:
            # TODO: OP_MODE에 config_mode 추가?
            # 현재는 Triage 일 경우 의도적으로 deepai.json이 없어서 except를 발생시킴
            self.device = os.environ['DEVICE'] if device is None else device
            self.gpu_index = os.environ['GPU_IDX'] if gpu_index is None else gpu_index
            self.init_models()

        self.result_path = Path('/deep/ai/app/data/result')
        self.original_path = Path('/deep/ai/app/data/original')

    def read_config(self, web_json):
        self.web_json = web_json
        self.write_version = True
        if 'visualization' in web_json['models']:
            self.web_vis = web_json['models']['visualization']
            if '@write_version' in self.web_vis:
                if self.web_vis['@write_version']['selected'] == 'Off':
                    self.write_version = False
        else:
            self.web_vis = None
        if 'secondary_capture' in web_json['models']:
            self.web_sc = web_json['models']['secondary_capture']
        else:
            self.web_sc = None
        if 'report' in web_json['models']:
            self.web_report = web_json['models']['report']
        elif ('advanced' in web_json) and ('report' in web_json['advanced']):
            self.web_report = web_json['advanced']['report']
        else:
            self.web_report = None
        if 'grayscale_softcopy_presentation_state' in web_json['models']:
            self.web_gsps = web_json['models']['grayscale_softcopy_presentation_state']
        elif ('advanced' in web_json) and ('grayscale_softcopy_presentation_state' in web_json['advanced']):        # TODO: 2D vs 3D Difference     --> 2D 와 합칠 경우 차이나는 부분도 합치는걸로 결정 (With AI 제품화팀)
            self.web_gsps = web_json['advanced']['grayscale_softcopy_presentation_state']
        else:
            self.web_gsps = None
        if ('advanced' in web_json) and ('use_modality' in web_json['advanced']):             # TODO: 2D vs 3D Difference
            self.web_use_modality = web_json['advanced']['use_modality']
        else:
            self.web_use_modality = None
        if 'intended_uses' in web_json['info']:
            self.web_intended_uses = web_json['info']['intended_uses']
        else:
            self.web_intended_uses = None
        if 'logo' in web_json:                                                                                      # TODO: 2D vs 3D Difference     --> 2D는 다른 위치에 logo의 default 값이 다른 곳에서 처리하도록 되어있어 정리 필요. 현재 모든 제품에서 logo에 대한 key 값은 없으므로 advanced로 이동 필요. 
            self.web_logo = web_json['logo']
            if "background_color" not in self.web_logo:
                self.web_logo["background_color"] = [0, 0, 0, 153]
        else:
            self.web_logo = {
                "write_logo": "On",
                "location": "PREVIOUS",
                "background": "Off",
                "background_color": [0, 0, 0, 153],
                "write_version": "On"
            }
            self.write_version = True
        if ('advanced' in web_json) and ('overlay_color' in web_json['advanced']):
            self.web_overlay_color = web_json['advanced']['overlay_color']
        else:
            self.web_overlay_color = { 'red': 255, 'green': 0, 'blue': 0 }
        logger.info(f'web_overlay_color: {self.web_overlay_color}')

        self.store_original = False
        self.store_result = False

        if ('advanced' in web_json) and ('store_original' in web_json['advanced']):
            if web_json['advanced']['store_original'].lower() == 'true':
                self.store_original = True
        if ('advanced' in web_json) and ('store_result' in web_json['advanced']):
            if web_json['advanced']['store_result'].lower() == 'true':
                self.store_result = True

        logger.info(f'store_original: {self.store_original}')
        logger.info(f'store_result: {self.store_result}')
        
    def init_models(self):
        raise NotImplementedError

    def make_report(self):
        raise NotImplementedError

    def get_study_and_info(self, input_path: Path):
        if input_path.is_file():
            scanned_studies = scan_file(input_path)
        else:
            scanned_studies = scan_directory(input_path)

        study_uids = [key for key in scanned_studies.keys()]

        logger.info(f'study_uids: {study_uids}')
        if len(study_uids) > 1:
            logger.error(f'Only 1 study supported, but includes {len(study_uids)} studies!')
            raise AssertionError
        study_uid = study_uids[0]

        study_paths = scanned_studies[study_uid]                        # study_paths 필요
        series_uids = [key for key in study_paths.keys()]               # series_uids 필요

        ds_init = pydicom.dcmread(study_paths[series_uids[0]][0], force=True)
        preferred_charsets = ''                                         # preferred_charsets 필요
        req_types = []                                                  # req_types 필요
        # TODO: 'FILE'이 들어가지 않는 output_mode중 output_path가 필요할 경우 수정 필요
        if 'FILE' in output_mode:
            if 'AI_REQUEST_TAG' in os.environ and os.environ['AI_REQUEST_TAG']:
                if os.environ['AI_REQUEST_TAG'] == 'Private3355':
                    if (0x3355, 0x1002) in ds_init:
                        try:
                            req_types = ds_init[0x3355, 0x1002].value.split('::')[-1].split('&')
                            preferred_charsets = ds_init[0x3355, 0x1003].value.split('::')[-1]
                        except:
                            logger.error("Invalid value in 'Private3355'!")
                            raise ValueError("Invalid value in 'Private3355'!")
                    else:
                        logger.error('There is no Private3355 tag!')
                        raise ValueError('There is no Private3355 tag!')
                elif os.environ['AI_REQUEST_TAG'] == 'StudyComments':
                    if (0x0032, 0x4000) in ds_init:
                        try:
                            _, req_types, preferred_charsets = ds_init.StudyComments.split('::')[2::2]
                            req_types = req_types.split('&')
                        except:
                            logger.error("Invalid value in 'StudyComments'!")
                            raise ValueError("Invalid value in 'StudyComments'!")
                    else:
                        logger.error('There is no StudyComments tag!')
                        raise ValueError('There is no StudyComments tag!')
                else:
                    raise KeyError("Invalid AI_REQUEST_TAG!")
            elif self.output_types:
                if 'sc' in self.output_types:
                    req_types.append('Secondary-Capture')
                if 'gsps' in self.output_types:
                    req_types.append('GSPS')
                if 'report' in self.output_types:
                    req_types.append('Report')
                if 'json' in self.output_types:
                    req_types.append('JSON')
                preferred_charsets = ''
            else:
                req_types = []
                if self.web_sc and self.web_sc['@create']['selected'] == 'On':
                    req_types.append('Secondary-Capture')
                if self.web_report and self.web_report['@create']['selected'] != 'Off':
                    req_types.append('Report')
                if self.web_gsps and self.web_gsps['@create']['selected'] == 'On':
                    req_types.append('GSPS')
                preferred_charsets = ''

        tags = {
            "patient_id": str(ds_init.get("PatientID", "")),
            "patient_name": str(ds_init.get("PatientName", "")),
            "patient_age": str(ds_init.get("PatientAge", "")),  # Not required, default is None
            "modality": str(ds_init.get("Modality", "")),
            "study_date": str(ds_init.get("StudyDate", "")),
            "study_time": str(ds_init.get("StudyTime", "")),
            "image_count": len(study_paths[series_uids[0]]),
            "study_uid": str(study_uid),
        }
        
        return study_paths, series_uids, preferred_charsets, req_types, tags

    def store_original_files(self, input_path: Path, api_key: str, study_uid: str) -> None:
        """
        web_json의 'store_original' 키 값이 'True'일 때, 모델의 인풋으로 들어온 DICOM 원본 디렉터리를
        복사 저장하는 메서드
        """
        self.original_path = Path('/deep/ai/app/data/original')
        store_dir = f'{self.original_path}/{api_key}/{study_uid}/'

        if not os.path.exists(store_dir):
            os.makedirs(store_dir)

        for item in os.listdir(input_path):
            if item == '.recieve-complete':    # '.recieve-complete' 파일은 제외
                continue

            source_item_path = os.path.join(input_path, item)
            store_item_path = os.path.join(store_dir, item)

            if os.path.isdir(source_item_path):
                shutil.copytree(source_item_path, store_item_path, dirs_exist_ok=True)
            else:    # '.recieve-complete' 등 파일은 제외
                shutil.copy2(source_item_path, store_item_path)
                continue
        
        logger.info('store_original_files() done')

    def add_logo_annotation(self, graphic_annotation_sequence, referenced_image, result_json):
        text_object_sequence = []
        self.add_logo_object(text_object_sequence, result_json)
        graphic_annotation = get_graphic_annotation_dataset(
            referenced_image_sequence=[referenced_image],
            layer_name='LOGO',
            text_object_sequence=text_object_sequence,
            graphic_object_sequence=[]
        )
        graphic_annotation_sequence.append(graphic_annotation)

    def make_gsps_dcm(
        self,
        ds,
        result_json,
        finding_sequences,
        displayed_area_selection_sequence,
        graphic_annotation_sequence,
        referenced_series_sequence,
        conversion_source_attributes_sequence,
        VOI_LUT_sequence,
        output_path,
        store_result_path,
        generate_dcm=None
        ):
        ds_gsps = make_dataset(ds, self.product_name, self.app_name, self.app_version, result_dcm_format='GSPS', use_uid=self.generated_uuid, preferred_charset=self.preferred_charsets)
        ds_gsps.SeriesNumber = self.web_gsps['series_number']['value']
        ds_gsps.GraphicLayerSequence = self.graphic_layer_sequence

        if len(finding_sequences):
            finding_sequences = sorted(finding_sequences, key=lambda seq: json.loads(seq[0x2021, 0x1009].value)['Probability'], reverse=True)

        if ds.PhotometricInterpretation == 'MONOCHROME1':
            ds_gsps.PresentationLUTShape = 'INVERSE'

        set_private_tags(ds_gsps, finding_sequences, self.app_name, self.app_version, self.product_name)
        ds_gsps.DisplayedAreaSelectionSequence = displayed_area_selection_sequence
        ds_gsps.GraphicAnnotationSequence = graphic_annotation_sequence
        ds_gsps.ReferencedSeriesSequence = referenced_series_sequence
        ds_gsps.ConversionSourceAttributesSequence = conversion_source_attributes_sequence
        ds_gsps.SoftcopyVOILUTSequence = VOI_LUT_sequence

        if 'Report' in result_json:
            js = result_json["Report"]
            ds_gsps.add_new((0x2021, 0x1010), 'UT', json.dumps(js))

        ds_gsps.save_as(str(output_path / 'gsps.dcm'))
        logger.info(str(output_path / 'gsps.dcm'))
        if self.store_result:
            ds_gsps.save_as(str(store_result_path / 'gsps.dcm'))
            logger.info(str(store_result_path / 'gsps.dcm'))

    def draw_logo(self, arr, width, height):
        mfont = ImageFont.truetype(os.path.join(medical_image.__path__[0], 'fonts/NotoSansCJKkr-Medium.otf'), int(height / 70))             # TODO: 2D version
        # mfont = ImageFont.truetype(os.path.join(medical_image.__path__[0], 'fonts/NotoSansCJKkr-Medium.otf'), int(height / 50))             # TODO: 3D version
        lfont = ImageFont.truetype(os.path.join(medical_image.__path__[0], 'fonts/NotoSansCJKkr-Medium.otf'), int(height / 50))

        image = Image.fromarray(arr.astype(np.uint8))
        img_editable = ImageDraw.Draw(image, 'RGBA')             # TODO: 2D version
        img_editable = ImageDraw.Draw(image)                     # TODO: 3D version

        msg = self.app_name.upper()
        if self.write_version:
            build_number = ''
            try:
                if ('advanced' in self.web_json) and ('show_build_number' in self.web_json['advanced']) and (self.web_json['advanced']['show_build_number'].lower() == 'true'):
                    build_number = '-' + os.environ['BUILD_NUMBER']
            except:
                pass

            msg += ' v' + self.app_version + build_number

        if self.web_logo['location'] == 'PREVIOUS':
            logo_w = self.logo.size[0] * height * 0.55 / 2472
            logo_h = self.logo.size[1] * height * 0.55 / 2472
            logo_x = (width - logo_w) / 2
            logo_y = height * 0.945 - logo_h

            font = lfont
            # mw, mh, _, _ = font.getbbox(msg)
            mw, mh = font.getsize(msg)
            shadow = int(mh / 20)
            mx = (width - mw) / 2
            my = logo_y - mh
        else:
            font = mfont
            # mw, mh, _, _ = font.getbbox(msg)
            mw, mh = font.getsize(msg)
            shadow = int(mh / 20)
            logo_w = mw
            logo_h = self.logo.size[1] * mw / self.logo.size[0]
            if self.web_logo['location'] == 'RU':
                logo_x = width - logo_w
                logo_y = 0
            elif self.web_logo['location'] == 'MU':
                logo_x = (width - logo_w) / 2
                logo_y = 0
            elif self.web_logo['location'] == 'ML':
                logo_x = (width - logo_w) / 2
                logo_y = height * 0.999 - logo_h - mh
            else:
                raise NotImplementedError
            mx = logo_x
            my = logo_y + logo_h
            w = mw

        h = logo_h + mh

        if self.web_logo["background"] == "On":
            if self.web_logo['location'] == 'PREVIOUS':
                if logo_w > mw:
                    x = logo_x
                    w = logo_w
                else:
                    x = mx
                    w = mw
                y = my
                graphic_data = np.array([
                    [x-w*0.025, y-h*0.025],
                    [x-w*0.025, y+h*1.025],
                    [x+w*1.025, y+h*1.025],
                    [x+w*1.025, y-h*0.025]
                ])
            else:
                x = logo_x
                y = logo_y
                if self.web_logo['location'] == 'RU':
                    graphic_data = np.array([
                        [x-w*0.025, y],
                        [x-w*0.025, y+h*1.1],
                        [x+w, y+h*1.1],
                        [x+w, y]
                    ])
                elif self.web_logo['location'] == 'MU':
                    graphic_data = np.array([
                        [x-w*0.025, y],
                        [x-w*0.025, y+h*1.1],
                        [x+w*1.025, y+h*1.1],
                        [x+w*1.025, y]
                    ])
                elif self.web_logo['location'] == 'ML':
                    graphic_data = np.array([
                        [x-w*0.025, y-h*0.1],
                        [x-w*0.025, y+h],
                        [x+w*1.025, y+h],
                        [x+w*1.025, y-h*0.1]
                    ])
                else:
                    raise NotImplementedError
            img_editable.polygon(
                [tuple(g) for g in graphic_data],
                fill=tuple(self.web_logo["background_color"])
            )
        paste_image(image, self.logo, logo_x, logo_y, logo_w, logo_h, 1)
        img_editable.text((mx + shadow, my + shadow), msg, (167, 167, 167), font)
        img_editable.text((mx, my), msg, (255, 255, 255), font)
        return np.array(image)

    def add_logo_object(self, text_object_sequence, result_json):
        if 'Height' in result_json: height = result_json['Height']
        else: height =  result_json['GraphicAnnotationSequence'][0]['Objects'][0]['Height']

        if 'Width' in result_json: width = result_json['Width']
        else: width =  result_json['GraphicAnnotationSequence'][0]['Objects'][0]['Width']

        mfont = ImageFont.truetype(os.path.join(medical_image.__path__[0], 'fonts/NotoSansCJKkr-Medium.otf'), int(height / 50))

        ## Logo
        msg = '[ D E E P N O I D ]'
        # w0, h0, _, _ = mfont.getbbox(msg)
        w0, h0 = mfont.getsize(msg)
        x = (width - w0) / 2
        y = height * 0.97 - h0
        # y = result_json['GraphicAnnotationSequence'][0]['Objects'][0]['Height'] * 0.97 - h0        # 3D version

        shadow = int(h0 / 20)
        text_style = get_text_style_dataset(
            color=[255, 255, 255],
            shadow_style='NORMAL',
            shadow_color=[255, 255, 255],
            shadow_offset=[shadow, shadow],
        )
        text_object_sequence.append(
            get_text_object_dataset(
                text_data=msg,
                text_style=text_style,
                bbox=[x, y, x + w0, y + h0]
            )
        )

        ## App
        msg = self.app_name.upper()
        build_number = ''
        try:
            if ('advanced' in self.web_json) and ('show_build_number' in self.web_json['advanced']) and (self.web_json['advanced']['show_build_number'].lower() == 'true'):
                build_number = '-' + os.environ['BUILD_NUMBER']
        except:
            pass
            
        if self.write_version:
            msg += ' v' + self.app_version + build_number

        # w1, h1, _, _ = mfont.getbbox(msg)
        w1, h1 = mfont.getsize(msg)
        x = (result_json['Width'] - w1) / 2
        # x = (result_json['GraphicAnnotationSequence'][0]['Objects'][0]['Width'] - w1) / 2                     # 3D version
        y = result_json['Height'] * 0.965 - h0 - h1
        # y = result_json['GraphicAnnotationSequence'][0]['Objects'][0]['Height'] * 0.965 - h0 - h1               # 3D version

        shadow = int(h1 / 20)
        text_style = get_text_style_dataset(
            color=[255, 255, 255],
            shadow_style='NORMAL',
            shadow_color=[255, 255, 255],
            shadow_offset=[shadow, shadow],
        )
        text_object_sequence.append(
            get_text_object_dataset(
                text_data=msg,
                text_style=text_style,
                bbox=[x, y, x + w1, y + h1]
            )
        )

    def dump_result_json(self, result_json, output_path, save_file_name):
        class ResultEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                # elif isinstance(obj, sitk.Image):
                #     logger.info(f'obj: {obj}')
                #     return save_sitk_image(obj, output_path)
                return json.JSONEncoder.default(self, obj)

        copied_result_json = copy.copy(result_json)
        if 'InputArray' in copied_result_json:
            del copied_result_json['InputArray']

        if 'SCDict' in copied_result_json:
            del copied_result_json['SCDict']

        del copied_result_json['SCArray']
        del copied_result_json['ReportArray']
        del copied_result_json['Skipping']
        
        if save_file_name is None:
            save_file_name = 'result'

        # TODO: 개선할 방법 없나?
        os.makedirs(output_path, exist_ok=True)
        if 'visualization' in copied_result_json:
            for key, images in copied_result_json['visualization'].items():
                copied_result_json['visualization'][key] = save_sitk_image(images, output_path, key)
                
        with open(f'{str(output_path)}/{save_file_name}.json', 'w', encoding='utf-8') as json_file:
            json.dump(copied_result_json, json_file, ensure_ascii=False, cls=ResultEncoder, indent=4)

    def insert_result_data(self, patient_study_id, result_json):
        if 'DB' not in output_mode:
            return
        
        # TODO: 모든 모델들에 TargetClassesProbs 항목 추가?
        if "TargetClassesProbs" in result_json: # chest는 현재 여러 질병이 묶여 나올 수도 있음에 따라 질병에 따른 확률 정리 dict으로 처리
            for disease_key, probability in result_json["TargetClassesProbs"].items():
                if probability > 0:
                    disease_code = crud.get_disease_code_by_disease_info(disease_key)
                    if disease_code:
                        insert_result = crud.insert_diagnosis_result(patient_study_id=patient_study_id,
                                                                probability=str(probability),
                                                                disease_code=disease_code,
                                                                succeeded="Y",
                                                                created_by="anonymousUser",
                                                                last_modified_by="anonymousUser"
                                                                )
        else: # 그 외의 경우는 GraphicAnnotationSequence에 따라 처리
            for sequence in result_json['GraphicAnnotationSequence']:
                probability = sequence['Probability']
                disease_code = crud.get_disease_code_by_disease_info(sequence["GraphicLayer"])
                if disease_code:
                    insert_result = crud.insert_diagnosis_result(patient_study_id=patient_study_id,
                                                            probability=str(probability),
                                                            disease_code=disease_code,
                                                            succeeded="Y",
                                                            created_by="anonymousUser",
                                                            last_modified_by="anonymousUser"
                                                            )

        if not insert_result:
            # DB Insert가 실패하는 경우, triage에서 failed 처리됨 
            logger.error(f'failed to insert diagnosis_result in {patient_study_id}: {len(result_json["GraphicAnnotationSequence"])} results')

    def load_encrypted_json(self, file_path: Path):
        with open(file_path, 'r', encoding='utf-8') as _file:
            loaded_json = _file.read()
        
        if op_mode == 'SECURED':
            decryted_json = Decrypt.decrypt_string(loaded_json)
            return json.loads(decryted_json)
        else:
            return json.loads(loaded_json)
            
    def load_encrypted_nii(self, result_root_path: Path, images: list):
        nii_files = []
        
        tmp_path = Path("/tmp") / "result"
        shutil.copytree(result_root_path, tmp_path)
        for image in images:
            file_path = tmp_path / image
            if op_mode == 'SECURED':
                Decrypt.decrypt_file(file_path)
            nii_file = sitk.ReadImage(file_path)
            nii_files.append(nii_file)
        
        shutil.rmtree(tmp_path)
        
        return nii_files

    def read_history(self, patient_id: str):
        if 'DB' not in output_mode:
            return {}
        
        result = {}
        root_path = Path("/deep/ai/app/data/")

        get_info = crud.get_study_uid_list(self.api_key, patient_id)
        if get_info is None:
            logger.info("That patient doesn't have any history")
            return result
            
        for info in get_info:
            if info["study_uid"] == self.study_uid:
                continue
                
            result_root_path = Path(root_path / "result")
            original_dir_path = Path(root_path / "original" / self.api_key / info["study_uid"])
            json_path = Path(result_root_path  / self.api_key / info["study_uid"] / 'result.json')

            if not (json_path.exists() and original_dir_path.exists()):
                continue
            load_json = self.load_encrypted_json(json_path)
            if "visualization" in load_json:
                for key, nii_files in load_json["visualization"].items():
                    load_nii = self.load_encrypted_nii(result_root_path, nii_files)
                    load_json["visualization"][key] = load_nii
            
            load_json["original_path"] = original_dir_path.resolve()
            result[info["datetime"].strftime("%Y%m%d%H%M%S")] = load_json

        return result

class DicomModule2D(DicomModule):
    def __init__(self, web_json, device=None, gpu_index=None):
        super().__init__(web_json, device, gpu_index)

    def run(self, web_json, input_path: Path, output_path, api_key, output_types=None):
        # Initializing all settings with deepai.json
        # TODO: 'FILE'이 들어가지 않는 output_mode중 output_path가 필요할 경우 수정 필요
        if 'FILE' in output_mode:
            self.read_config(web_json)

        # SDK
        self.output_types = output_types

        # Initializing study_uid for encrypting
        self.study_uid = None

        self.patient_study_id = None
        self.api_key = api_key

        # Initializing input study
        if 'DB' in output_mode:
            self.patient_study_id = crud.make_row_with_api(api_key)
        study_paths, series_uids, preferred_charsets, req_types, tags = self.get_study_and_info(input_path)

        # TODO: 'FILE'이 들어가지 않는 output_mode중 output_path가 필요할 경우 수정 필요
        if 'FILE' in output_mode and self.store_original:    # Store original DICOM input
            self.store_original_files(input_path=input_path, api_key=api_key, study_uid=tags["study_uid"])

        self.study_uid = tags["study_uid"]
        if 'DB' in output_mode:
            crud.update_patient_study(patient_study_id=self.patient_study_id,
                                      patient_id = tags["patient_id"],
                                      patient_name = tags["patient_name"],
                                      patient_age = tags["patient_age"],
                                      modality = tags["modality"],
                                      image_count = tags["image_count"],
                                      study_date = tags["study_date"],
                                      study_time = tags["study_time"],
                                      api_key = api_key,
                                      study_uid = tags["study_uid"],
                                      created_by = "anonymousUser",
                                      last_modified_by = "anonymousUser"
            )
        
        store_result_path = Path(self.result_path / api_key / tags["study_uid"])

        if output_mode == 'DB':
            # 이전 분석 결과의 graphic_layer_sequence 값이 누적되지 않도록 초기화!
            self.graphic_layer_sequence = self.default_graphic_layer_sequence
            for series_uid in series_uids:
                series_paths = study_paths[series_uid]
                referenced_image_sequence = []
                finding_sequence = []
                for path in series_paths:
                    ds = pydicom.dcmread(str(path), force=True)
                    result_json = self.run_models(ds)

            self.dump_result_json(result_json, store_result_path, 'result')
            self.insert_result_data(self.patient_study_id, result_json)
        # TODO: 'FILE'이 들어가지 않는 output_mode중 output_path가 필요할 경우 수정 필요
        elif 'FILE' in output_mode:
            self.preferred_charsets = preferred_charsets.split('\\')[-1]
            logger.info(f'req_types: {req_types}')
            logger.info(f'preferred_charsets: {self.preferred_charsets}')

            # Initializing output format
            referenced_series_sequence = []
            displayed_area_selection_sequence = []
            graphic_annotation_sequence = []
            VOI_LUT_sequence = []
            finding_sequences = []
            conversion_source_attributes_sequence = []
            # 이전 분석 결과의 graphic_layer_sequence 값이 누적되지 않도록 초기화!
            self.graphic_layer_sequence = self.default_graphic_layer_sequence
            self.generated_uuid = uuid.uuid4().int
            save_file_name = None
            for series_uid in series_uids:
                series_paths = study_paths[series_uid]
                referenced_image_sequence = []
                finding_sequence = []
                for path in series_paths:
                    ds = pydicom.dcmread(str(path), force=True)
                    result_json = self.run_models(ds)
                    if 'Height' not in result_json: result_json['Height'] = ds.Rows
                    if 'Width' not in result_json: result_json['Width'] = ds.Columns
                    if self.output_types:
                        # SDK라는 의미!!!
                        save_file_name = path.stem

                        if 'JSON' in req_types:
                            # SDK이면서 JSON 출력이 필요할 경우
                            self.dump_result_json(result_json, output_path, save_file_name)
                    elif 'DB' in output_mode:
                        self.dump_result_json(result_json, store_result_path, 'result')
                        self.insert_result_data(self.patient_study_id, result_json)

                    if 'dump_result_json' in web_json['advanced'] and web_json['advanced']['dump_result_json'].lower() == 'true':
                        result_json_path = Path('/deep/ai/app/data/results') / output_path.name
                        logger.warning(f'result_json_path = {result_json_path}')
                        result_json_path.mkdir(parents=True, exist_ok=True)
                        self.dump_result_json(result_json, result_json_path, series_uid)

                    if ('Skipping' in result_json and not result_json['Skipping']) or (not 'Skipping'in result_json):
                        referenced_image = get_referenced_image_dataset(ds)
                        referenced_image_sequence.append(referenced_image)
                        conversion_source_attributes_sequence.append(referenced_image)
                        displayed_area_selection_sequence.append(get_displayed_area_selection(ds, [referenced_image]))
                        VOI_LUT_sequence.append(get_VOI_LUT_dataset(ds, [referenced_image]))

                        finding_sequence += self.update_gsps(ds, graphic_annotation_sequence, referenced_image, result_json)
                        self.add_logo_annotation(graphic_annotation_sequence, referenced_image, result_json)

                        if ('Secondary-Capture' in req_types) and (('CreateNormal' not in result_json) or ('SC' in result_json['CreateNormal'])):
                            self.make_sc_dcm(ds, result_json, output_path, save_file_name, store_result_path)

                        if ('Report' in req_types) and (('CreateNormal' not in result_json) or ('Report' in result_json['CreateNormal'])):
                            report_array = self.make_report(ds, result_json)                # Newly added function for making report array. Defined in main.py for each app functions.
                            # report_array = result_json['ReportArray']
                            if type(report_array) == np.ndarray:
                                ds_rp = make_dataset(ds, self.product_name, self.app_name, self.app_version, result_dcm_format='RP', use_uid=self.generated_uuid, preferred_charset=self.preferred_charsets, use_modality=self.web_use_modality)
                                copy_dicom_private_tags(ds, ds_rp)
                                set_private_tags(ds_rp, finding_sequence, self.app_name, self.app_version, self.product_name)
                                ds_rp.PixelData = report_array.tobytes()
                                ds_rp.Rows = report_array.shape[0]
                                ds_rp.Columns = report_array.shape[1]
                                if '@append_series_number' in self.web_report:
                                    if self.web_report['@append_series_number']['selected'] == 'On' and 'SeriesNumber' in ds and ds.SeriesNumber:
                                        ds_rp.SeriesNumber = self.web_report['series_number']['value'] + str(ds.SeriesNumber)
                                    else:
                                        ds_rp.SeriesNumber = self.web_report['series_number']['value']
                                elif 'series_number' in self.web_report:
                                    ds_rp.SeriesNumber = self.web_report['series_number']['value']
                                else:
                                    ds_rp.SeriesNumber = '888882' + (str(ds.SeriesNumber) if 'SeriesNumber' in ds and ds.SeriesNumber else '')
                                if 'Report' in result_json:
                                    if len(result_json["GraphicAnnotationSequence"]):
                                        finding_pair = self.get_finding_pair(result_json)
                                    else:
                                        finding_pair = [("No Finding", 0.00)]
                                    js = result_json["Report"]
                                    ds_rp.add_new((0x2021, 0x1010), 'UT', json.dumps(js))
                                    ds_rp.add_new((0x2021, 0x1009), 'UT', json.dumps({
                                        'Name': ', '.join([i for i, j in finding_pair if i != '']),
                                        'Probability': str(max([j for i, j in finding_pair if i != ''])),
                                    }))
                                
                                if save_file_name is None:
                                    save_file_name = ds_rp.SOPInstanceUID
                                else:
                                    save_file_name = f'{save_file_name}-REPORT'

                                ds_rp.save_as(str(output_path / (save_file_name + '.dcm')))
                                logger.info(str(output_path / (save_file_name + '.dcm')))
                                if self.store_result:
                                    ds_rp.save_as(str(store_result_path / (save_file_name + '.dcm')))
                                    logger.info(str(store_result_path / (save_file_name + '.dcm')))
                            else:
                                logger.info('report array has wrong type : {}'.format(type(report_array)))
                                break

                if len(finding_sequence):
                    finding_sequences += finding_sequence
                    referenced_series_sequence.append(get_referenced_series_dataset(series_uid, referenced_image_sequence))

            if len(finding_sequences):
                if ('GSPS' in req_types) and (('CreateNormal' not in result_json) or ('GSPS' in result_json['CreateNormal'])):
                    self.make_gsps_dcm(
                        ds,
                        result_json,
                        finding_sequences,
                        displayed_area_selection_sequence,
                        graphic_annotation_sequence,
                        referenced_series_sequence,
                        conversion_source_attributes_sequence,
                        VOI_LUT_sequence,
                        output_path,
                        store_result_path
                    )

    def insert_dicom_data(self, ds, result_json):
        if 'DB' not in output_mode:
            return
        
        dicom_tags = ['PatientID', 'PatientName', 'PatientAge', 'Modality', 'StudyDate', 'StudyTime']
        for dicom_tag in dicom_tags:
            if not hasattr(ds, dicom_tag):
                setattr(ds, dicom_tag, '')

        ptx_probability = '0.00'
        pef_probability = '0.00'
        # logger.info(f'result_json["GraphicAnnotationSequence"]: {result_json["GraphicAnnotationSequence"]}')
        for graphic_annotation in result_json["GraphicAnnotationSequence"]:
            if graphic_annotation['GraphicLayer'] == 'PTX' and ptx_probability == '0.00':
                ptx_probability = str(float(graphic_annotation['Probability']))
            elif graphic_annotation['GraphicLayer'] == 'PEF' and pef_probability == '0.00':
                pef_probability = str(float(graphic_annotation['Probability']))
     
        crud.insert_patient_study(str(ds.PatientID), str(ds.PatientName), str(ds.PatientAge), str(pef_probability), ptx_probability, ds.Modality, '1', ds.StudyDate, ds.StudyTime, 'Y')

    def make_origin_dcm(self, ds, result_json, output_path):                                    # TODO: 2D vs 3D Difference
        finding_pair = []
        if len(result_json["GraphicAnnotationSequence"]):
            finding_pair = self.get_finding_pair(result_json)

        original_study_desc = ds.StudyDescription if 'StudyDescription' in ds else ''
        if len(finding_pair):
            ds.StudyDescription = '*[' + ','.join([i for i, j in finding_pair]) + ']* ' + original_study_desc
        else:
            ds.StudyDescription = '[] ' + original_study_desc

        try:
            series_idx = int(ds.SeriesNumber)
        except:
            series_idx = 1
        try:
            instance_idx = int(ds.InstanceNumber)
        except:
            instance_idx = 1

        sc_idx=1
        use_uid = self.generated_uuid

        replace_str = '2'
        if ds.StudyInstanceUID[:1] == '2':
            replace_str = '3'

        ds.StudyInstanceUID = replace_str + ds.StudyInstanceUID[1:]

        ds.SeriesInstanceUID = '2.25.{:d}.{:d}.{:d}'.format(series_idx, sc_idx, use_uid)
        ds.SOPInstanceUID = '2.25.{:d}.{:d}.{:d}.{:d}'.format(series_idx, sc_idx, instance_idx, use_uid)
        ds.SeriesNumber = str(series_idx)
        ds.InstanceNumber = str(instance_idx)

        if self.web_use_modality:
            ds.Modality = self.web_use_modality

        ds.save_as(str(output_path) + f'/{ds.StudyInstanceUID}.dcm')

    def get_finding_pair(self, result_json):                                                # TODO: 2D vs 3D Difference
        finding_names, finding_pair = [], []
        for graphic_annotation in result_json["GraphicAnnotationSequence"]:
            # name = graphic_annotation['GraphicLayer'].replace(',', '')
            name = graphic_annotation['GraphicLayer']
            if ',' in name:
                names = name.split(', ')
                # for n in names:
                #     finding_names.append(n)
            else:
                names = [name]
                # finding_names.append(name)
            probability = float(graphic_annotation['Probability'])

            for name in names:
                if name in finding_names:
                    continue
                else:
                    finding_names.append(name)
                    finding_pair.append((name, probability))
        finding_pair.sort(key=lambda x: x[1], reverse=True)
        return finding_pair

    def make_sc_dcm(self, ds, result_json, output_path, save_file_name, store_result_path):
        # assert (type(result_json['InputArray']) == np.ndarray) or type(result_json['SCArray']) == np.ndarray
        referenced_image = get_referenced_image_dataset(ds)

        ds_sc = make_dataset(ds, self.product_name, self.app_name, self.app_version, result_dcm_format='SC', use_uid=self.generated_uuid, preferred_charset=self.preferred_charsets, use_modality=self.web_use_modality)
        ds_sc.WindowCenter = 128
        ds_sc.WindowWidth = 255
        if '@append_series_number' in self.web_sc:
            if self.web_sc['@append_series_number']['selected'] == 'On' and 'SeriesNumber' in ds and ds.SeriesNumber:
                ds_sc.SeriesNumber = self.web_sc['series_number']['value'] + str(ds.SeriesNumber)
            else:
                ds_sc.SeriesNumber = self.web_sc['series_number']['value']
        elif 'series_number' in self.web_sc:
            ds_sc.SeriesNumber = self.web_sc['series_number']['value']
        else:
            ds_sc.SeriesNumber = '888881' + (str(ds.SeriesNumber) if 'SeriesNumber' in ds and ds.SeriesNumber else '')

        assert result_json['Height'] == ds.Rows
        assert result_json['Width'] == ds.Columns
        if self.output_types:
            result_json['InputArray'] = read_dcm(ds)
            sc_array = self.make_sc_array(result_json)
        else:
            if type(result_json["SCArray"]) == np.ndarray:
                sc_array = result_json["SCArray"]
            else:
                # assert "InputArray" not in result_json
                assert result_json["SCArray"] == None
                result_json['InputArray'] = read_dcm(ds)
                sc_array = self.make_sc_array(result_json)

        output_w = result_json["Width"]
        output_h = result_json["Height"]
        pixel_data = self.draw_logo(sc_array, output_w, output_h)
        if 'SizeFactor' in result_json:
            output_w *= result_json['SizeFactor']
            output_w = int(output_w)
            output_h *= result_json['SizeFactor']
            output_h = int(output_h)
            pixel_data = cv2.resize(pixel_data, (output_w, output_h))
        ds_sc.Rows = output_h
        ds_sc.Columns = output_w
        ds_sc.PixelData = pixel_data.tobytes()

        finding_sequence = []
        # finding_names, finding_pair = [], []
        if len(result_json["GraphicAnnotationSequence"]):
            # for graphic_annotation in result_json["GraphicAnnotationSequence"]:
            #     name = graphic_annotation['GraphicLayer'].replace(',', '')
            #     probability = float(graphic_annotation['Probability'])
            #     if name in finding_names:
            #         continue
            #     else:
            #         finding_names.append(name)
            #         finding_pair.append((name, probability))
            # finding_pair.sort(key=lambda x:x[1], reverse=True)
            finding_pair = self.get_finding_pair(result_json)

            dataset = Dataset()
            dataset.add_new((0x2021, 0x0010), 'LO', 'DEEPNOID')
            dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                'Name': ', '.join([i for i, j in finding_pair if i != '']),
                'Probability': str(max([j for i, j in finding_pair if i != ''])),
            }))
            dataset.ReferencedImageSequence = [referenced_image]
            finding_sequence.append(dataset)

        else:
            dataset = Dataset()
            dataset.add_new((0x2021, 0x0010), 'LO', 'DEEPNOID')
            dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                'Name': 'No Finding',
                'Probability': 0.0000
            }))
            dataset.ReferencedImageSequence = [referenced_image]
            finding_sequence = [dataset]

        set_private_tags(ds_sc, finding_sequence, self.app_name, self.app_version, self.product_name)
        if type(result_json['ImageComments']) == str:
            ds_sc.ImageComments = result_json['ImageComments']

        if 'Report' in result_json:
            js = result_json["Report"]
            ds_sc.add_new((0x2021, 0x1010), 'UT', json.dumps(js))

        if save_file_name is None:
            save_file_name = ds_sc.SOPInstanceUID
        else:
            save_file_name = f'{save_file_name}-SC'

        ds_sc.save_as(str(output_path / (save_file_name + '.dcm')))
        logger.info(str(output_path / (save_file_name + '.dcm')))
        if self.store_result:
            ds_sc.save_as(str(store_result_path / (save_file_name + '.dcm')))
            logger.info(str(store_result_path / (save_file_name + '.dcm')))

    def make_sc_array(self, result_json):
        sfont = ImageFont.truetype(os.path.join(medical_image.__path__[0], 'fonts/NotoSansCJKkr-Medium.otf'), int(result_json['Height'] / 90))
        image = Image.fromarray(result_json['InputArray'].astype(np.uint8))
        img_editable = ImageDraw.Draw(image)

        ## GraphicObject
        contour_image_array = np.array(image)
        for graphic_annotation in result_json["GraphicAnnotationSequence"]:
            heatmap_str = graphic_annotation['Heatmap']
            if heatmap_str != None:
                heatmap_image = Image.open(BytesIO(base64.b64decode(heatmap_str)))
                heatmap = np.array(heatmap_image)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGRA2BGR)
                contour_image_array = cv2.addWeighted(contour_image_array, 1, heatmap, float(self.web_vis['opacity']['value']), 0)
                heatmap_image.close()

            for graphic_object in graphic_annotation['GraphicObjectSequence']:
                graphic_data = np.array(graphic_object['graphic_data'], np.int32)
                line_color = graphic_object['color'] if 'color' in graphic_object else (255, 255, 255)
                if 'thickness' in graphic_object:
                    line_thickness = graphic_object['thickness']
                else:
                    line_thickness = max(int(min(result_json['Width'], result_json['Height']) / 500), 1)

                if line_thickness == -1:
                    if 'fill_opacity' in graphic_object:
                        opacity = graphic_object['fill_opacity']
                    else:
                        opacity = 1.0
                    line_color += (int(255 * opacity),)

                    image = Image.fromarray(contour_image_array)
                    img_editable = ImageDraw.Draw(image, 'RGBA')
                    img_editable.polygon(
                        [tuple(g) for g in graphic_data],
                        fill=line_color
                    )
                    contour_image_array = np.array(image)
                else:
                    if ('shadow_color' not in graphic_object) or (graphic_object['shadow_color'] != 'OFF'):
                        shadow_color = graphic_object['shadow_color'] if 'shadow_color' in graphic_object else (0, 0, 0)
                        contour_image_array = cv2.polylines(contour_image_array, [graphic_data], False, shadow_color, int(line_thickness * 2.5))
                    contour_image_array = cv2.polylines(contour_image_array, [graphic_data], False, line_color, line_thickness)

                if 'arrow_point' in graphic_object:
                    start_point, end_point = graphic_object['arrow_point']
                    contour_image_array = cv2.arrowedLine(contour_image_array, end_point, start_point, (0, 0, 0), line_thickness+10, tipLength=0.2)
                    contour_image_array = cv2.arrowedLine(contour_image_array, end_point, start_point, line_color, line_thickness+5, tipLength=0.2)

        image = Image.fromarray(contour_image_array)
        img_editable = ImageDraw.Draw(image, 'RGBA')
        ## TextObject
        for graphic_annotation in result_json["GraphicAnnotationSequence"]:
            for text_object in graphic_annotation['TextObjectSequence']:
                if 'font_type' in text_object:
                    font_type = text_object['font_type']
                else:
                    font_type = 'Medium'
                if 'font_scale' in text_object:
                    font_scale = text_object['font_scale']
                else:
                    # font_scale = int(result_json['Height'] / 70)
                    font_scale = int(result_json['Height'] / 40)


                font = ImageFont.truetype(os.path.join(medical_image.__path__[0], f'fonts/NotoSansCJKkr-{font_type}.otf'), font_scale)
                text_color = text_object['color'] if 'color' in text_object else (255, 255, 255)
                msg = text_object['text_data']
                x, y, w, h = text_object['bbox']
                if 'arrow_point' in graphic_annotation['GraphicObjectSequence']:
                    img_editable.text((x-2, y-2), msg, (0, 0, 0), font)
                    img_editable.text((x-2, y+2), msg, (0, 0, 0), font)
                    img_editable.text((x+2, y-2), msg, (0, 0, 0), font)
                    img_editable.text((x+2, y+2), msg, (0, 0, 0), font)
                    img_editable.text((x, y), msg, text_color, font)
                elif ('shadow_color' not in text_object) or (text_object['shadow_color'] != 'OFF'):
                    # mw, mh, _, _ = font.getbbox(msg)
                    mw, mh = font.getsize(msg)
                    shadow = int(mh / 20)
                    shadow_color = text_object['shadow_color'] if 'shadow_color' in text_object else (167, 167, 167)
                    img_editable.text((x + shadow, y + shadow), msg, shadow_color, font)
                img_editable.text((x, y), msg, text_color, font)

        ## Colorbar
        heatmaps = [g['Heatmap'] for g in result_json["GraphicAnnotationSequence"]]
        # if len(heatmaps) != heatmaps.count(""):
        if len(heatmaps) != heatmaps.count(None):
            w, h = self.colorbar.size
            w *= result_json['Height'] * 0.4 / 2472 * 1000 / 1757
            h *= result_json['Height'] * 0.4 / 2472 * 1000 / 1757
            x = result_json['Width'] * 0.03
            y = result_json['Height'] * 0.985 - h
            paste_image(image, self.colorbar, x, y, w, h, 1)
            # hw, hh, _, _ = sfont.getbbox('High')
            hw, hh = sfont.getsize('High')
            img_editable.text((x, y - hh), 'Low', (255, 255, 255), sfont)
            img_editable.text((x + w - hw, y - hh), 'High', (255, 255, 255), sfont)

        return np.array(image)

    def update_gsps(self, ds, graphic_annotation_sequence, referenced_image, result_json):
        width = int(ds.Columns)
        height = int(ds.Rows)

        if len(result_json["GraphicAnnotationSequence"]):
            finding_sequence = []
            for graphic_annotation in result_json["GraphicAnnotationSequence"]:
                graphic_annotation_sequence.append(
                    get_graphic_annotation_dataset(
                        referenced_image_sequence=[referenced_image],
                        layer_name=graphic_annotation['GraphicLayer'],
                        text_object_sequence=get_text_object_dcm(graphic_annotation["TextObjectSequence"]),
                        graphic_object_sequence=get_graphic_object_dcm(graphic_annotation["GraphicObjectSequence"], width, height)
                    )
                )
                dataset = Dataset()
                dataset.add_new((0x2021, 0x0010), 'LO', 'DEEPNOID')
                if graphic_annotation['Heatmap'] == None:
                    heatmap_str = ''
                    opacity = 0.5
                else:
                    heatmap_str = graphic_annotation['Heatmap']
                    opacity = self.web_vis['opacity']['value']
                dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                    'Name': graphic_annotation['GraphicLayer'],
                    'Probability': float(graphic_annotation['Probability']),
                    'Heatmap': heatmap_str,
                    'BlendingRatio': float(opacity)
                }))
                dataset.ReferencedImageSequence = [referenced_image]
                finding_sequence.append(dataset)

        else:
            dataset = Dataset()
            dataset.add_new((0x2021, 0x0010), 'LO', 'DEEPNOID')
            dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                'Name': 'No Finding',
                'Probability': 0.0000,
                'Heatmap': '',
                'BlendingRatio': float(0.5)
            }))
            dataset.ReferencedImageSequence = [referenced_image]
            finding_sequence = [dataset]
        return sorted(finding_sequence, key=lambda seq: json.loads(seq[0x2021, 0x1009].value)['Probability'], reverse=True)

    def make_gsps_dcm(
        self,
        ds,
        result_json,
        finding_sequences,
        displayed_area_selection_sequence,
        graphic_annotation_sequence,
        referenced_series_sequence,
        conversion_source_attributes_sequence,
        VOI_LUT_sequence,
        output_path,
        store_result_path
        ):
        ds_gsps = make_dataset(ds, self.product_name, self.app_name, self.app_version, result_dcm_format='GSPS', use_uid=self.generated_uuid, preferred_charset=self.preferred_charsets)
        ds_gsps.SeriesNumber = self.web_gsps['series_number']['value']
        ds_gsps.GraphicLayerSequence = self.graphic_layer_sequence

        if len(finding_sequences):
            finding_sequences = sorted(finding_sequences, key=lambda seq: json.loads(seq[0x2021, 0x1009].value)['Probability'], reverse=True)

        if ds.PhotometricInterpretation == 'MONOCHROME1':
            ds_gsps.PresentationLUTShape = 'INVERSE'

        set_private_tags(ds_gsps, finding_sequences, self.app_name, self.app_version, self.product_name)
        ds_gsps.DisplayedAreaSelectionSequence = displayed_area_selection_sequence
        ds_gsps.GraphicAnnotationSequence = graphic_annotation_sequence
        ds_gsps.ReferencedSeriesSequence = referenced_series_sequence
        ds_gsps.ConversionSourceAttributesSequence = conversion_source_attributes_sequence
        ds_gsps.SoftcopyVOILUTSequence = VOI_LUT_sequence

        if 'Report' in result_json:
            js = result_json["Report"]
            ds_gsps.add_new((0x2021, 0x1010), 'UT', json.dumps(js))

        ds_gsps.save_as(str(output_path / 'GSPS.dcm'))
        logger.info(str(output_path / 'GSPS.dcm'))
        if self.store_result:
            ds_gsps.save_as(str(store_result_path / 'GSPS.dcm'))
            logger.info(str(store_result_path / 'GSPS.dcm'))


class DicomModule3D(DicomModule):
    def __init__(self, web_json, logger=None):
        self.logger = logger
        super().__init__(web_json, logger=logger)

    def run(self, web_json, input_path: Path, output_path, api_key, output_types=None):
        # Initializing all settings with deepai.json
        self.read_config(web_json)

        # SDK
        self.output_types = output_types

        # Initializing study_uid for encrypting
        self.study_uid = None

        self.patient_study_id = None
        self.api_key = api_key

        # Initializing input study
        if 'DB' in output_mode:
            self.patient_study_id = crud.make_row_with_api(api_key)
        
        split_enhanced_dicom_path = None
        # Enhanced DICOM이면 분리해서 저장하고, input_path를 해당 경로로 변경
        if is_enhanced_dicom(input_path):
            split_enhanced_dicom_path = split_enhanced_dicom(input_path)
            input_path = split_enhanced_dicom_path

        study_paths, series_uids, preferred_charsets, req_types, tags = self.get_study_and_info(input_path)

        if self.store_original:    # Store original DICOM input
            self.store_original_files(input_path=input_path, api_key=api_key, study_uid=tags["study_uid"])
        
        self.study_uid = tags["study_uid"]
        if 'DB' in output_mode:
            crud.update_patient_study(patient_study_id=self.patient_study_id,
                                      patient_id = tags["patient_id"],
                                      patient_name = tags["patient_name"],
                                      patient_age = tags["patient_age"],
                                      modality = tags["modality"],
                                      image_count = tags["image_count"],
                                      study_date = tags["study_date"],
                                      study_time = tags["study_time"],
                                      api_key = api_key,
                                      study_uid = tags["study_uid"],
                                      created_by = "anonymousUser",
                                      last_modified_by = "anonymousUser"
            )
        
        store_result_path = Path(self.result_path / api_key / tags["study_uid"])
        self.preferred_charsets = preferred_charsets.split('\\')[-1]
        logger.info(f'req_types: {req_types}')
        logger.info(f'preferred_charsets: {self.preferred_charsets}')

        # Initializing output format
        referenced_series_sequence = []
        displayed_area_selection_sequence = []
        graphic_annotation_sequence = []
        VOI_LUT_sequence = []
        finding_sequences = []
        conversion_source_attributes_sequence = []
        # 이전 분석 결과의 graphic_layer_sequence 값이 누적되지 않도록 초기화!
        self.graphic_layer_sequence = self.default_graphic_layer_sequence
        self.generated_uuid = uuid.uuid4().int

        for series_uid in series_uids:
            # 현재 3D모델은 모두 series가 input받는 구조로 만들어져 있음
            # input이 study이거나.. instance인 모델이 발생하게 되면 추가 작업이 필요함.
            series_paths = study_paths[series_uid]
            referenced_image_sequence = []
            ds_dict = {'path': series_paths[0].parent}
            for path in series_paths:
                ds = pydicom.dcmread(str(path), force=True)
                ds_dict[int(ds.InstanceNumber)] = ds

                if output_mode == 'DB':
                    # 굳이 결과 파일 생성이 필요 없음!
                    continue
                
                referenced_image = get_referenced_image_dataset(ds)
                referenced_image_sequence.append(referenced_image)
                conversion_source_attributes_sequence.append(referenced_image)
                displayed_area_selection_sequence.append(get_displayed_area_selection(ds, [referenced_image]))
                VOI_LUT_sequence.append(get_VOI_LUT_dataset(ds, [referenced_image]))

            result_json = self.run_models(ds_dict, req_types)

            if 'Height' not in result_json: result_json['Height'] = ds.Rows
            if 'Width' not in result_json: result_json['Width'] = ds.Columns

            if 'DB' in output_mode:
                self.dump_result_json(result_json, store_result_path, 'result')
                self.insert_result_data(self.patient_study_id, result_json)
            
            if output_mode == 'DB':
                # 굳이 결과 파일 생성이 필요 없음!
                continue
            
            # finding_sequence = self.get_finding_sequence(referenced_image_sequence, result_json)
            graphic_annotation_sequence = []
            finding_sequences = []
            finding_sequence = []
            for ins in ds_dict:
                if not type(ins) == str:
                    ds = ds_dict[ins]
                    referenced_image = get_referenced_image_dataset(ds)
                    finding_sequence += self.update_private_tags(ds, graphic_annotation_sequence, referenced_image, result_json)
            if len(finding_sequence):
                finding_sequences += finding_sequence
                finding_sequences = sorted(finding_sequences, key=lambda seq: json.loads(seq[0x2021, 0x1009].value)['Probability'], reverse=True)
            if not result_json['Skipping']:
                if ('Secondary-Capture' in req_types) and (('CreateNormal' not in result_json) or ('SC' in result_json['CreateNormal'])):
                    self.make_sc_dcm(ds_dict, finding_sequences, result_json, output_path, store_result_path, split_enhanced_dicom_path)

                if ('SCDict' in result_json) and (('CreateNormal' not in result_json) or ('MIP' in result_json['CreateNormal'])):
                    sc_dict = result_json['SCDict']
                    # 뇌동맥류에서 MIP 결과 생성을 위하여 임시적으로 생성됨
                    for key in sc_dict:
                        assert type(key) == int
                        series_number = sc_dict[key]['SeriesNumber']
                        for i, sc_array in enumerate(sc_dict[key]['SCArray']):
                            ds_i  = copy.deepcopy(ds)
                            ds_i.InstanceNumber = i+1
                            ds_sc = make_dataset(ds_i, self.product_name, self.app_name, self.app_version, result_dcm_format='MIP', use_uid=self.generated_uuid, preferred_charset=self.preferred_charsets, sc_idx=key, use_modality=self.web_use_modality)
                            copy_dicom_private_tags(ds_i, ds_sc)
                            set_private_tags(ds_sc, finding_sequences, self.app_name, self.app_version, self.product_name)
                            ds_sc.Columns = sc_array.shape[1]
                            ds_sc.Rows = sc_array.shape[0]
                            # ds_sc.PixelData = self.draw_logo(sc_array, sc_array.shape[1], sc_array.shape[0]).tobytes()
                            if self.web_logo["write_logo"] == 'On':
                                ds_sc.PixelData = self.draw_logo(sc_array, sc_array.shape[1], sc_array.shape[0]).tobytes()
                            else:
                                ds_sc.PixelData = sc_array.tobytes()
                            # ds_sc.PixelData = sc_array.tobytes()
                            ds_sc.SeriesNumber = series_number
                            if type(result_json['ImageComments']) == str: ds_sc.ImageComments = result_json['ImageComments']
                            if 'Report' in result_json:
                                js = result_json["Report"]
                                ds_sc.add_new((0x2021, 0x1010), 'UT', json.dumps(js))
                            ds_sc.save_as(str(output_path / (ds_sc.SOPInstanceUID + '.dcm')))
                            if self.store_result:
                                ds_sc.save_as(str(store_result_path / (ds_sc.SOPInstanceUID + '.dcm')))

            if ('Report' in req_types) and (('CreateNormal' not in result_json) or ('Report' in result_json['CreateNormal'])):
                report_array = result_json['ReportArray']
                if type(report_array) == list:  # 2 페이지 이상의 Report
                    for i, report_page_array in enumerate(report_array):
                        ds_i  = copy.deepcopy(ds)
                        ds_i.InstanceNumber = i+1
                        ds_rp = make_dataset(ds_i, self.product_name, self.app_name, self.app_version, result_dcm_format='RP', use_uid=self.generated_uuid, preferred_charset=self.preferred_charsets, use_modality=self.web_use_modality)
                        copy_dicom_private_tags(ds_i, ds_rp)
                        set_private_tags(ds_rp, finding_sequences, self.app_name, self.app_version, self.product_name)
                        ds_rp.PixelData = report_page_array.tobytes()
                        ds_rp.Rows = report_page_array.shape[0]
                        ds_rp.Columns = report_page_array.shape[1]
                        if '@append_series_number' in self.web_report:
                            if self.web_report['@append_series_number']['selected'] == 'On' and 'SeriesNumber' in ds and ds.SeriesNumber:
                                ds_rp.SeriesNumber = self.web_report['series_number']['value'] + str(ds.SeriesNumber)
                            else:
                                ds_rp.SeriesNumber = self.web_report['series_number']['value']
                        elif 'series_number' in self.web_report:
                            ds_rp.SeriesNumber = self.web_report['series_number']['value']
                        else:
                            ds_rp.SeriesNumber = '888882' + (str(ds.SeriesNumber) if 'SeriesNumber' in ds and ds.SeriesNumber else '')
                        
                        if type(result_json['ImageComments']) == str: ds_rp.ImageComments = result_json['ImageComments']
                        if 'Report' in result_json:
                            js = result_json["Report"]
                            ds_rp.add_new((0x2021, 0x1010), 'UT', json.dumps(js))
                        ds_rp.save_as(str(output_path / (ds_rp.SOPInstanceUID + '.dcm')))
                        if self.store_result:
                            ds_rp.save_as(str(store_result_path / (ds_rp.SOPInstanceUID + '.dcm')))

                elif type(report_array) == np.ndarray:
                    ds_rp = make_dataset(ds, self.product_name, self.app_name, self.app_version, result_dcm_format='RP', use_uid=self.generated_uuid, preferred_charset=self.preferred_charsets, use_modality=self.web_use_modality)
                    copy_dicom_private_tags(ds, ds_rp)
                    set_private_tags(ds_rp, finding_sequences, self.app_name, self.app_version, self.product_name)
                    ds_rp.PixelData = report_array.tobytes()
                    ds_rp.Rows = report_array.shape[0]
                    ds_rp.Columns = report_array.shape[1]
                    if '@append_series_number' in self.web_report:
                        if self.web_report['@append_series_number']['selected'] == 'On' and 'SeriesNumber' in ds and ds.SeriesNumber:
                            ds_rp.SeriesNumber = self.web_report['series_number']['value'] + str(ds.SeriesNumber)
                        else:
                            ds_rp.SeriesNumber = self.web_report['series_number']['value']
                    elif 'series_number' in self.web_report:
                        ds_rp.SeriesNumber = self.web_report['series_number']['value']
                    else:
                        ds_rp.SeriesNumber = '888883' + (str(ds.SeriesNumber) if 'SeriesNumber' in ds and ds.SeriesNumber else '')
                    if type(result_json['ImageComments']) == str: ds_rp.ImageComments = result_json['ImageComments']
                    if 'Report' in result_json:
                        js = result_json["Report"]
                        ds_rp.add_new((0x2021, 0x1010), 'UT', json.dumps(js))

                    ds_rp.save_as(str(output_path / (ds_rp.SOPInstanceUID + '.dcm')))
                    if self.store_result:
                        ds_rp.save_as(str(store_result_path / (ds_rp.SOPInstanceUID + '.dcm')))
                            
                else:
                    logger.info('report array has wrong type : {}'.format(type(report_array)))
                    break

        if output_mode == 'DB':
            # 굳이 GSPS 결과 파일 생성이 필요 없음!
            # 나머지 SC나 MIP, Report는 위의 for 문 안에서 처리됨
            return 
        
        if ('GSPS' in req_types) and (('CreateNormal' not in result_json) or ('GSPS' in result_json['CreateNormal'])):
            # Enhanced DICOM 파일일 경우 GSPS 생성하지 않음
            if split_enhanced_dicom_path:
                logger.warning("Grayscale Softcopy Presentation State (GSPS) is not supported for Enhanced DICOM")
                return

            referenced_series_sequence = []
            displayed_area_selection_sequence = []
            graphic_annotation_sequence = []
            VOI_LUT_sequence = []
            finding_sequences = []
            conversion_source_attributes_sequence = []
            referenced_image_sequence = []
            finding_sequence = []
            for ins in ds_dict:
                if not type(ins) == str:
                    ds = ds_dict[ins]
                    referenced_image = get_referenced_image_dataset(ds)
                    referenced_image_sequence.append(referenced_image)
                    conversion_source_attributes_sequence.append(referenced_image)
                    displayed_area_selection_sequence.append(get_displayed_area_selection(ds, [referenced_image]))
                    VOI_LUT_sequence.append(get_VOI_LUT_dataset(ds, [referenced_image]))
                    
                    finding_sequence += self.update_private_tags(ds, graphic_annotation_sequence, referenced_image, result_json, True)
                    self.add_logo_annotation(graphic_annotation_sequence, referenced_image, result_json)

            if len(finding_sequence):
                finding_sequences += finding_sequence
                finding_sequences = sorted(finding_sequences, key=lambda seq: json.loads(seq[0x2021, 0x1009].value)['Probability'], reverse=True)
                referenced_series_sequence.append(get_referenced_series_dataset(ds.SeriesInstanceUID, referenced_image_sequence))

            self.make_gsps_dcm(
                ds,
                result_json,
                finding_sequences,
                displayed_area_selection_sequence,
                graphic_annotation_sequence,
                referenced_series_sequence,
                conversion_source_attributes_sequence,
                VOI_LUT_sequence,
                output_path,
                store_result_path
            )
        
        if split_enhanced_dicom_path:
            # input_path에 지정된, 분리했던 frames 폴더 삭제
            shutil.rmtree(input_path, ignore_errors=True)

    def update_private_tags(self, ds, graphic_annotation_sequence, referenced_image, result_json, is_gsps=False): ####### ds
        width = int(ds.Columns)
        height = int(ds.Rows)
        if len(result_json["GraphicAnnotationSequence"]):
            finding_sequence = []
            for graphic_annotation in result_json["GraphicAnnotationSequence"]:
                extracted_object = extract_object_by_instance_number(graphic_annotation['Objects'], ds.InstanceNumber) #######
                if extracted_object: #########
                    finding_sequence = []
                    
                    graphic_annotation_sequence.append(
                        get_graphic_annotation_dataset(
                            referenced_image_sequence=[referenced_image],
                            layer_name=graphic_annotation['GraphicLayer'],
                            text_object_sequence=get_text_object_dcm(extracted_object["TextObjectSequence"]),
                            graphic_object_sequence=get_graphic_object_dcm(extracted_object["GraphicObjectSequence"], width, height)
                        )
                    )
                    dataset = Dataset()
                    dataset.add_new((0x2021, 0x0010), 'LO', 'DEEPNOID')
                    if extracted_object['Heatmap'] == None:
                        heatmap_str = ''
                        opacity = 0.5
                    else:
                        heatmap_str = extracted_object['Heatmap']
                        opacity = 0.5 ####### origin : self.web_vis['opacity']['value']
                    if is_gsps:
                        dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                            'Name': graphic_annotation['GraphicLayer'],
                            'Probability': float(graphic_annotation['Probability']),
                            'Heatmap': heatmap_str,
                            'BlendingRatio': float(opacity)
                        }))
                    else:
                        dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                            'Name': graphic_annotation['GraphicLayer'],
                            'Probability': float(graphic_annotation['Probability'])
                        }))
                    dataset.ReferencedImageSequence = [referenced_image]
                    finding_sequence.append(dataset)

                else:
                    dataset = Dataset()
                    dataset.add_new((0x2021, 0x0010), 'LO', 'DEEPNOID')
                    if is_gsps:
                        dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                            'Name': 'No Finding',
                            'Probability': 0.0000,
                            'Heatmap': '',
                            'BlendingRatio': float(0.5)
                        }))
                    else:
                        dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                            'Name': 'No Finding',
                            'Probability': 0.0000
                        }))
                    dataset.ReferencedImageSequence = [referenced_image]
                    finding_sequence = [dataset]
        else:
            dataset = Dataset()
            dataset.add_new((0x2021, 0x0010), 'LO', 'DEEPNOID')
            if is_gsps:
                dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                    'Name': 'No Finding',
                    'Probability': 0.0000,
                    'Heatmap': '',
                    'BlendingRatio': float(0.5)
                }))
            else:
                dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                    'Name': 'No Finding',
                    'Probability': 0.0000
                }))
            dataset.ReferencedImageSequence = [referenced_image]
            finding_sequence = [dataset]
        return sorted(finding_sequence, key=lambda seq: json.loads(seq[0x2021, 0x1009].value)['Probability'], reverse=True)

    def make_sc_dcm(self, ds_dict, finding_sequence, result_json, output_path, store_result_path, split_enhanced_dicom_path=None):
        # assert (type(result_json['InputArray']) == np.ndarray) or type(result_json['SCArray']) == np.ndarray
        instance_numbers = sorted([key for key in ds_dict.keys() if type(key) == int])
        if type(result_json["SCArray"]) == np.ndarray:
            result_sc_array = result_json["SCArray"]
            assert len(result_sc_array) == len(instance_numbers)

        object_numbers = {}
        for i, graphic_annotation in enumerate(result_json["GraphicAnnotationSequence"]):
            objects = graphic_annotation['Objects']
            _ins = [obj['instance_number'] for obj in objects]
            for _in in _ins:
                if _in in object_numbers:
                    object_numbers[_in][i] = _ins.index(_in)
                else:
                    object_numbers[_in] = {i: _ins.index(_in)}

        for j, instance_number in enumerate(instance_numbers):
            ds = ds_dict[instance_number]
            ds_sc = make_dataset(ds, self.product_name, self.app_name, self.app_version, result_dcm_format='SC', use_uid=self.generated_uuid, preferred_charset=self.preferred_charsets, use_modality=self.web_use_modality)
            ds_sc.WindowCenter = 128
            ds_sc.WindowWidth = 255
            if '@append_series_number' in self.web_sc:
                if self.web_sc['@append_series_number']['selected'] == 'On' and 'SeriesNumber' in ds and ds.SeriesNumber:
                    ds_sc.SeriesNumber = self.web_sc['series_number']['value'] + str(ds.SeriesNumber)
                else:
                    ds_sc.SeriesNumber = self.web_sc['series_number']['value']
            elif 'series_number' in self.web_sc:
                ds_sc.SeriesNumber = self.web_sc['series_number']['value']
            else:
                ds_sc.SeriesNumber = '888881' + (str(ds.SeriesNumber) if 'SeriesNumber' in ds and ds.SeriesNumber else '')

            objects = []
            if instance_number in object_numbers:
                for i in object_numbers[instance_number]:
                    objects.append(result_json["GraphicAnnotationSequence"][i]["Objects"][object_numbers[instance_number][i]])

            if type(result_json["SCArray"]) == np.ndarray:
                sc_array = result_json["SCArray"][j]
                width = int(ds.Columns)
                height = int(ds.Rows)
            else:
                sc_array, width, height = self.make_sc_array(ds, objects, result_json)

            if self.web_logo["write_logo"] == 'On':
                ds_sc.PixelData = self.draw_logo(sc_array, width, height).tobytes()
            else:
                ds_sc.PixelData = sc_array.tobytes()

            ds_sc.Rows = height
            ds_sc.Columns = width
            # if len(objects):
            #     ds_sc.Rows = objects[0]['Height']
            #     ds_sc.Columns = objects[0]['Width']
            # else:
            #     ds_sc.Rows = int(ds.Rows)
            #     ds_sc.Columns = int(ds.Columns)

            set_private_tags(ds_sc, finding_sequence, self.app_name, self.app_version, self.product_name)
            if type(result_json['ImageComments']) == str:
                ds_sc.ImageComments = result_json['ImageComments']

            if 'SetTags' in result_json:
                for tag, value in result_json['SetTags']:
                    setattr(ds_sc, tag, value)
                    # if ds_sc.__contains__(tag):
                    #     assert ds_sc[tag].value == value
                    # else:
                    #     setattr(ds_sc, tag, value)

            if 'Report' in result_json:
                js = result_json["Report"]
                ds_sc.add_new((0x2021, 0x1010), 'UT', json.dumps(js))

            ds_sc.save_as(str(output_path / (ds_sc.SOPInstanceUID + '.dcm')))
            if self.store_result:
                ds_sc.save_as(str(store_result_path / (ds_sc.SOPInstanceUID + '.dcm')))
            # logger.info(str(output_path / (ds_sc.SOPInstanceUID + '.dcm')))

        if split_enhanced_dicom_path:
            if not (('split_edcm_sc' in self.web_json['advanced']) and (self.web_json['advanced']['split_edcm_sc'].lower() == 'true')):
                merge_dicom_to_enhanced(output_path)

    def get_finding_sequence(self, referenced_image_sequence, result_json):
        # TODO: 2D처럼 graphic_annotation_sequence 업데이트 하는 내용 추가 (for GSPS)
        if len(result_json["GraphicAnnotationSequence"]):
            finding_sequence = []
            for graphic_annotation in result_json["GraphicAnnotationSequence"]:
                dataset = Dataset()
                dataset.add_new((0x2021, 0x0010), 'LO', 'DEEPNOID')
                dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                    'Name': graphic_annotation['GraphicLayer'],
                    'Probability': float(graphic_annotation['Probability']),
                }))
                dataset.ReferencedImageSequence = referenced_image_sequence
                finding_sequence.append(dataset)
        else:
            dataset = Dataset()
            dataset.add_new((0x2021, 0x0010), 'LO', 'DEEPNOID')
            dataset.add_new((0x2021, 0x1009), 'UT', json.dumps({
                'Name': 'No Finding',
                'Probability': 0.0000
            }))
            dataset.ReferencedImageSequence = referenced_image_sequence
            finding_sequence = [dataset]
        return sorted(finding_sequence, key=lambda seq: json.loads(seq[0x2021, 0x1009].value)['Probability'], reverse=True)

    def make_sc_array(self, ds, objects, result_json):
        width = result_json["Width"]
        height = result_json["Height"]
        # if len(objects):
        #     height = objects[0]['Height']
        #     width = objects[0]['Width']
        # else:
        #     height = int(ds.Rows)
        #     width = int(ds.Columns)
        image = Image.fromarray(read_dcm(ds).astype(np.uint8))
        img_editable = ImageDraw.Draw(image)

        for obj in objects:
            ## GraphicObject
            contour_image_array = np.array(image)
            for graphic_object in obj['GraphicObjectSequence']:
                graphic_data = np.array(graphic_object['graphic_data'], np.int32)
                line_color = graphic_object['color'] if 'color' in graphic_object else (255, 255, 255)
                if 'thickness' in graphic_object:
                    line_thickness = graphic_object['thickness']
                else:
                    line_thickness = max(int(min(width, height) / 500), 1)

                if line_thickness == -1:
                    if 'fill_opacity' in graphic_object:
                        opacity = graphic_object['fill_opacity']
                    else:
                        opacity = 1.0
                    line_color += (int(255 * opacity),)

                    image = Image.fromarray(contour_image_array)
                    img_editable = ImageDraw.Draw(image, 'RGBA')
                    img_editable.polygon(
                        [tuple(g) for g in graphic_data],
                        fill=line_color
                    )
                    contour_image_array = np.array(image)
                else:
                    if ('shadow_color' not in graphic_object) or (graphic_object['shadow_color'] != 'OFF'):
                        shadow_color = graphic_object['shadow_color'] if 'shadow_color' in graphic_object else (0, 0, 0)
                        contour_image_array = cv2.polylines(contour_image_array, [graphic_data], False, shadow_color, 10)
                    contour_image_array = cv2.polylines(contour_image_array, [graphic_data], False, line_color, max(int(min(width, height) / 500), 1))


            image = Image.fromarray(contour_image_array)
            img_editable = ImageDraw.Draw(image)
            ## TextObject
            for text_object in obj['TextObjectSequence']:
                if 'font_type' in text_object:
                    font_type = text_object['font_type']
                else:
                    font_type = 'Medium'
                if 'font_scale' in text_object:
                    font_scale = text_object['font_scale']
                else:
                    font_scale = int(height / 70)

                font = ImageFont.truetype(os.path.join(medical_image.__path__[0], f'fonts/NotoSansCJKkr-{font_type}.otf'), font_scale)

                text_color = text_object['color'] if 'color' in text_object else (255, 255, 255)
                msg = text_object['text_data']
                x, y, _, _ = text_object['bbox']
                if ('shadow_color' not in text_object) or (text_object['shadow_color'] != 'OFF'):
                    # mw, mh, _, _ = font.getbbox(msg)
                    mw, mh = font.getsize(msg)
                    shadow = int(mh / 20)
                    shadow_color = text_object['shadow_color'] if 'shadow_color' in text_object else (167, 167, 167)
                    img_editable.text((x + shadow, y + shadow), msg, shadow_color, font)
                img_editable.text((x, y), msg, text_color, font)

        return np.array(image), width, height

    def draw_logo(self, arr, width, height):
        mfont = ImageFont.truetype(os.path.join(medical_image.__path__[0], 'fonts/NotoSansCJKkr-Medium.otf'), int(height / 50))

        image = Image.fromarray(arr.astype(np.uint8))
        img_editable = ImageDraw.Draw(image)

        ## Logo
        logo_w = self.logo.size[0] * height * 0.55 / 2472
        logo_h = self.logo.size[1] * height * 0.55 / 2472
        x = (width - logo_w) / 2
        y = height * 0.945 - logo_h
        paste_image(image, self.logo, x, y, logo_w, logo_h, 1)

        ## App Info
        msg = self.app_name.upper()
        build_number = ''
        try:
            if ('advanced' in self.web_json) and ('show_build_number' in self.web_json['advanced']) and (self.web_json['advanced']['show_build_number'].lower() == 'true'):
                build_number = '-' + os.environ['BUILD_NUMBER']
        except:
            pass
            
        if self.write_version:
            msg += ' v' + self.app_version + build_number

        mw, mh = mfont.getsize(msg)
        x = (width - mw) / 2
        y = height * 0.940 - logo_h - mh
        shadow = int(mh / 20)
        img_editable.text((x + shadow, y + shadow), msg, (167, 167, 167), mfont)
        img_editable.text((x, y), msg, (255, 255, 255), mfont)

        return np.array(image)

    def add_logo_object(self, text_object_sequence, result_json):

        # TODO: [BYK] result_json의 Object의 height가 실제 원본 dicom의 임의의 slice와 dimension이 다를까요? 같다면 result_json['Height']로 변경 가능할지 검토 필요 (DicomModule2D 참고 대략 line 770)
            # TODO: [Ans by BYK] Object의 height는 실제 원본 dicom의 모든 slice의 height와 같습니다. 현재는 instance 단위로 object를 부여하고 있으나, result_json['Height']로 변경 가능합니다.
        mfont = ImageFont.truetype(os.path.join(medical_image.__path__[0], 'fonts/NotoSansCJKkr-Medium.otf'), int(result_json['Height'] / 50))
        ## Logo
        msg = '[ D E E P N O I D ]'
        # w0, h0, _, _ = mfont.getbbox(msg)
        w0, h0 = mfont.getsize(msg)
        x = (result_json['Width'] - w0) / 2           # TODO: [BYK] result_json의 Object의 height가 실제 원본 dicom의 임의의 slice와 dimension이 다를까요? 같다면 result_json['Height']로 변경 가능할지 검토 필요 (DicomModule2D 참고 대략 line 770) - [Ans by BYK]: 상기 답변과 같음
        y = result_json['Height'] * 0.97 - h0         # TODO: [BYK] result_json의 Object의 height가 실제 원본 dicom의 임의의 slice와 dimension이 다를까요? 같다면 result_json['Height']로 변경 가능할지 검토 필요 (DicomModule2D 참고 대략 line 770) - [Ans by BYK]: 상기 답변과 같음
        shadow = int(h0 / 20)
        text_style = get_text_style_dataset(
            color=[255, 255, 255],
            shadow_style='NORMAL',
            shadow_color=[255, 255, 255],
            shadow_offset=[shadow, shadow],
        )
        text_object_sequence.append(
            get_text_object_dataset(
                text_data=msg,
                text_style=text_style,
                bbox=[x, y, x + w0, y + h0]
            )
        )

        msg = self.app_name      # TODO: [BYK] upper로 설정하는 별도 이유가 있을까요? - [Ans by BYK]: 기존에 설정되어있던 그대로 사용했었으나, 확인결과 web_json에서 model_name만 대문자로 설정하면 없애도 상관없을듯.
        build_number = ''                # TODO: [HJY] 2D에도 build number에 대한 조건 추가시 정상 작동 여부 확인 필요
        try:
            if ('advanced' in self.web_json) and ('show_build_number' in self.web_json['advanced']) and (self.web_json['advanced']['show_build_number'].lower() == 'true'):
                build_number = '-' + os.environ['BUILD_NUMBER']
        except:
            pass
            
        if self.write_version:
            msg += ' v' + self.app_version + build_number

        # w1, h1, _, _ = mfont.getbbox(msg)
        w1, h1 = mfont.getsize(msg)
        x = (result_json['Width'] - w1) / 2 ########
        y = result_json['Height'] * 0.965 - h0 - h1 #######
        shadow = int(h1 / 20)
        text_style = get_text_style_dataset(
            color=[255, 255, 255],
            shadow_style='NORMAL',
            shadow_color=[255, 255, 255],
            shadow_offset=[shadow, shadow],
        )
        text_object_sequence.append(
            get_text_object_dataset(
                text_data=msg,
                text_style=text_style,
                bbox=[x, y, x + w1, y + h1]
            )
        )

    def make_gsps_dcm(      # TODO: [BYK] 해당 코드는 DicomModule3D - run 함수에서 사용되고 있으며 대략 991-1003줄에서 사용되고 있습니다. return된 ds_gsps를 받아와서 slice별로 저장만 처리해주고 있는데 해당 내용을 이 함수내에서 처리하도록 변경 가능할지 확인 부탁드립니다. (DicomModule2D 참고)
        self,
        ds,
        result_json,
        finding_sequences,
        displayed_area_selection_sequence,
        graphic_annotation_sequence,
        referenced_series_sequence,
        conversion_source_attributes_sequence,
        VOI_LUT_sequence,
        output_path,
        store_result_path
        ):
        ds_gsps = make_dataset(ds, self.product_name, self.app_name, self.app_version, result_dcm_format='GSPS', use_uid=self.generated_uuid, preferred_charset=self.preferred_charsets, intended_uses=self.web_intended_uses) # TODO: [BYK] Hard coding된 제품명, 모델명, 버전명 등을 class attribute으로 변경할 수 있을지 검토 필요 (DicomModule2D 참고) - [Ans by BYK]: 수정완료
        ds_gsps.SeriesNumber = self.web_gsps['series_number']['value'] # TODO: [BYK] test 필요
        ds_gsps.GraphicLayerSequence = self.graphic_layer_sequence # TODO: [BYK] test 필요
        # ds_gsps.GraphicLayerSequence = [        # TODO: [BYK] DicomModule2D와 같이 self.graphic_layer_sequence로 변경 가능할지 검토 필요 (DicomMoudle2D 참고)
        #         get_graphic_layer_dataset('LOGO', 0),
        #         get_graphic_layer_dataset('NO FINDING', 1),
        #         get_graphic_layer_dataset('Cerebral Aneurysm', 2) ###### 원래 main.py에서 default_graphic_layer_sequence 업데이트함.
        #     ]
        
        # if len(finding_sequences):
        #     finding_sequences = sorted(finding_sequences, key=lambda seq: json.loads(seq[0x2021, 0x1009].value)['Probability'], reverse=True)

        if ds.PhotometricInterpretation == 'MONOCHROME1':
            ds_gsps.PresentationLUTShape = 'INVERSE'

        set_private_tags(ds_gsps, finding_sequences, self.app_name, self.app_version, self.product_name) ######       # TODO: [BYK] Hard coding된 제품명, 모델명, 버전명 등을 class attribute으로 변경할 수 있을지 검토 필요 (DicomModule2D 참고) - [Ans by BYK]: 수정완료
        ds_gsps.DisplayedAreaSelectionSequence = displayed_area_selection_sequence
        ds_gsps.GraphicAnnotationSequence = graphic_annotation_sequence
        ds_gsps.ReferencedSeriesSequence = referenced_series_sequence
        ds_gsps.ConversionSourceAttributesSequence = conversion_source_attributes_sequence
        ds_gsps.SoftcopyVOILUTSequence = VOI_LUT_sequence

        if 'Report' in result_json:
            js = result_json["Report"]
            ds_gsps.add_new((0x2021, 0x1010), 'UT', json.dumps(js))
        
        ds_gsps.save_as(str(output_path / f'{ds_gsps.SOPInstanceUID}.dcm'))
        if self.store_result:
            ds_gsps.save_as(str(store_result_path / f'{ds_gsps.SOPInstanceUID}.dcm'))
        # logger.info(str(output_path / f'{ds_gsps.SOPInstanceUID}.dcm'))
        
        # return ds_gsps  # TODO: [BYK] return 형태가 아니라 함수 내에서 flag (함수의 parameter를 기준으로) 바로 저장하는 코드로 변경 가능한지 확인 필요 (DicomModule2D 대략.. line 260-262 참고)


class DicomModuleHisto(object):
    def __init__(self, web_json, logger=None):
        self.read_config(web_json)
        self.device = os.environ['DEVICE']
        self.gpu_index = os.environ['GPU_IDX']
        self.init_models()

        self.product_name = web_json["product_name"] if web_json["product_name"] else 'TEST'
        self.app_name = web_json["model_name"] if web_json["model_name"] else 'TEST'
        self.app_version = web_json["info"]["udi"]["udi_pi"] if web_json["info"]["udi"]["udi_pi"] else 'TEST'

        self.default_graphic_layer_sequence = [
            get_graphic_layer_dataset('LOGO', 0),
            get_graphic_layer_dataset('NO FINDING', 1)
        ]
        self.graphic_layer_sequence = []

        self.logo = Image.open(os.path.join(medical_image.__path__[0], 'logo/deepnoid.png'))
        self.colorbar = Image.open(os.path.join(medical_image.__path__[0], 'logo/colorbar.png'))
        # self.preferred_character_set not initilized in HISTO                          # TODO: 2D vs 3D Difference

    def read_config(self, web_json):
        self.web_json = web_json
        self.write_version = True
        if 'visualization' in web_json['models']:
            self.web_vis = web_json['models']['visualization']
            if '@write_version' in self.web_vis:
                if self.web_vis['@write_version']['selected'] == 'Off':
                    self.write_version = False
        else:
            self.web_vis = None
        # No initilization for secondary_capture, report, GSPS, advnaced for HISTO          # TODO: 2D vs 3D Difference

    def init_models(self):
        raise NotImplementedError

    def make_report(self):
        raise NotImplementedError

    def run(self, web_json, input_path: Path, output_path):
        if ('advanced' in web_json) and ('#slide_exts' in web_json['advanced']) and ('checked' in web_json['advanced']['#slide_exts']):
            slide_exts = web_json['advanced']['#slide_exts']['checked']
        else:
            slide_exts = ["mrxs", "svs", "ndpi", "tif", "tiff"]
        
        for f in os.listdir(input_path):
            if f.split('.')[1] in slide_exts:
                slide_name = f.split('.')[0]
                slide_ext = f.split('.')[1]
        
        logger.info(f'Processing: {slide_name}.{slide_ext}')
        self.run_models(input_path, output_path, web_json, slide_name, slide_exts)
