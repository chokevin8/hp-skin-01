import os
from pathlib import Path
import cv2
import copy
import json
import uuid
import base64
import pydicom
import tempfile
import numpy as np
from datetime import datetime
from pydicom.dataset import Dataset
from pydicom.valuerep import DSfloat
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
from pydicom.pixel_data_handlers.util import convert_color_space, apply_color_lut
from pydicom.pixel_data_handlers.util import apply_voi_lut

import SimpleITK as sitk

# DEEP:HISTO
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import xml.etree.ElementTree as ET
import xml.dom.minidom

import warnings
warnings.filterwarnings("ignore", message="Invalid value for VR CS:")

def save_sitk_image(images, output_path: Path, key: int):
    file_paths = []
    for idx, image in enumerate(images):
        file_name = str(key) + "_" + str(idx) + ".nii.gz"
        file_path = output_path / file_name
        sitk.WriteImage(image, file_path)
        
        relative_path = str("/".join(file_path.parts[-3:]))
        file_paths.append(relative_path)
    
    return file_paths


def encoding_heatmap(mask, thres=0.5):
    assert mask.max() <= 1
    assert mask.min() >= 0

    grad_cam = mask
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(grad_cam - thres, 0, 1) / (1 - thres)), cv2.COLORMAP_JET)
    back_index = np.where(grad_cam < thres)
    heatmap[back_index] = 0
    b, g, r = cv2.split(heatmap)
    a = np.ones(b.shape, b.dtype) * 255
    a[back_index] = 0
    img_bgra = cv2.merge([b, g, r, a])
    tmp_png_path = tempfile.gettempdir() + '/' + str(uuid.uuid4()) + '.png'
    cv2_imwrite(tmp_png_path, img_bgra)
    with open(tmp_png_path, "rb") as _png_file:
        heatmap_encoded = base64.b64encode(_png_file.read()).decode()
    os.remove(tmp_png_path)
    return heatmap_encoded


def cv2_imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as exc:
        print(exc)
        return False


def rgb2Lab(color):
    color_Lab = cv2.cvtColor(np.array([[color]]).astype(np.uint8), cv2.COLOR_RGB2Lab).squeeze()
    return [color_Lab[0] * 256, color_Lab[1] * 256, color_Lab[2] * 256]


def paste_image(report, cam, x, y, w, h, scale):
    x1 = int(x * scale)
    y1 = int(y * scale)
    w = int(w * scale)
    h = int(h * scale)
    x2 = x1 + w
    y2 = y1 + h
    paste_image = cam.resize((w, h))
    if cam.mode == 'RGBA':
        report.paste(paste_image, [x1, y1, x2, y2], paste_image)
    else:
        report.paste(paste_image, [x1, y1, x2, y2])


def validate_window_values(ds):
    # Get Pixel Representation and Bits Stored
    pixel_representation = getattr(ds, "PixelRepresentation", 0)
    bits_stored = getattr(ds, "BitsStored", 16)
    
    # Determine valid range
    if pixel_representation == 0:  # Unsigned
        min_value = 0
        max_value = (2 ** bits_stored) - 1
    else:  # Signed
        min_value = -(2 ** (bits_stored - 1))
        max_value = (2 ** (bits_stored - 1)) - 1

    # Get Window Center & Width (can be multi-valued)
    window_center = getattr(ds, "WindowCenter", None)
    window_width = getattr(ds, "WindowWidth", None)
    
    # Ensure they are lists for multi-value handling
    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = list(map(float, window_center))
    elif window_center is not None:
        window_center = [float(window_center)]
    
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = list(map(float, window_width))
    elif window_width is not None:
        window_width = [float(window_width)]
    
    # Validate values
    # print() 호출 시 3D 모델에서는 너무 많은 로그가 출력되어 주석 처리
    if window_center and window_width:
        for wc, ww in zip(window_center, window_width):
            if not (min_value <= wc <= max_value):
                # print(f"Invalid WindowCenter: {wc} (Valid Range: {min_value} to {max_value})")
                return False
            elif wc == min_value:
                # print(f"WindowCenter is equal to min_value {min_value}, ignoring...")
                return False

            if ww <= 0:
                # print(f"Invalid WindowWidth: {ww} (Must be positive)")
                return False
            elif ww == 0:
                # print(f"WindowWidth is equal to min_value 0, ignoring...")
                return False
        return True
    else:
        # print("WindowCenter or WindowWidth not found in DICOM file.")
        return False


def read_dcm(ds):
    arr = ds.pixel_array
    mode = ds.PhotometricInterpretation
    if 'YBR_FULL' in mode:
        ori_image_array = convert_color_space(arr, mode, 'RGB')
    elif mode == 'RGB':
        ori_image_array = convert_color_space(arr, mode, 'RGB')
    elif mode == 'PALETTE COLOR':
        ori_image_array = apply_color_lut(arr, ds)
    elif 'MONOCHROME' in mode:
        try:
            if ds.__contains__('RescaleIntercept') and ds.__contains__('RescaleSlope'):
                intercept = ds.RescaleIntercept
                slope = ds.RescaleSlope
            else:
                intercept = 0
                slope = 1
            out = slope * arr + intercept
        except:
            out = arr

        if validate_window_values(ds):
            if type(ds.WindowCenter) == DSfloat:
                wc = ds.WindowCenter
                ww = ds.WindowWidth
            else:
                wc = ds.WindowCenter[0]
                ww = ds.WindowWidth[0]

            wl = int(wc - 0.5 - (ww - 1) / 2)
            wu = int(wc - 0.5 + (ww - 1) / 2)

            out[np.where(out <= wl)] = wl
            out[np.where(out >= wu)] = wu

            ori_image_array = (out - wl) / (wu - wl)

        elif ds.__contains__('VOILUTSequence'):
            out = apply_voi_lut(out, ds)
            ori_image_array = out / ds.VOILUTSequence[0].LUTDescriptor[0]

        else:
            # raise KeyError("No 'WindowWidth' nor 'WindowCenter' nor 'VOILUTSequence' in dataset!")
            out = np.asarray(out, dtype=np.float32)
            ori_image_array = cv2.normalize(out, None, 0, 1, cv2.NORM_MINMAX)

        if mode == 'MONOCHROME1':
            ori_image_array = 1 - ori_image_array

        ori_image_array *= 255

    else:
        raise KeyError("Invalid 'PhotometricInterpretation' in dataset!")

    if ori_image_array.ndim != 3:
        ori_image_array = np.stack([ori_image_array] * 3, axis=-1)
    if ori_image_array.shape[-1] == 4:
        ori_image_array = ori_image_array[..., :3]

    return ori_image_array


def scan_file(input_path):
    study_instance_uids = {}
    ds = pydicom.dcmread(str(input_path), force=True)
    study_uid = ds.StudyInstanceUID
    series_uid = ds.SeriesInstanceUID
    study_instance_uids.update(
        {
            study_uid: {
                series_uid: [input_path]
            }
        }
    )
    return study_instance_uids


def scan_directory(input_path, suffix='.dcm'):
    paths = sorted(list(input_path.glob('*' + suffix)))
    study_instance_uids = {}
    for path in paths:
        ds = pydicom.dcmread(str(path), force=True)
        study_uid = ds.StudyInstanceUID
        series_uid = ds.SeriesInstanceUID
        if study_uid in study_instance_uids:
            if series_uid in study_instance_uids[study_uid]:
                study_instance_uids[study_uid][series_uid].append(path)
            else:
                study_instance_uids[study_uid].update(
                    {
                        series_uid: [path]
                    }
                )

        else:
            study_instance_uids.update(
                {
                    study_uid: {
                        series_uid: [path]
                    }
                }
            )
    return study_instance_uids


def make_dataset(input_dicom, product_name, app_name, app_version, result_dcm_format='SC', use_uid=None, preferred_charset='', sc_idx=1, use_modality=None, intended_uses=None):
    ds = Dataset()
    copy_dicom_tags(ds, copy.deepcopy(input_dicom), result_dcm_format)
    set_dicom_tags(ds, product_name, app_name, app_version, result_dcm_format, use_modality, intended_uses)
    if preferred_charset:
        convert_charset(ds, preferred_charset)
    set_time(ds, result_dcm_format)
    set_uids(ds, input_dicom, use_uid, result_dcm_format, sc_idx)
    ds.preamble = input_dicom.preamble
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    return ds


def convert_charset(dicom, preferred_charset):
    python_encode_type = pydicom.charset.convert_encodings(dicom.SpecificCharacterSet)[0]
    python_decode_type = pydicom.charset.convert_encodings(preferred_charset)[0]
    for tag in ['InstitutionName', 'PatientName', 'StudyDescription']:
        if dicom.__contains__(tag):
            raw_value = dicom[tag].value
            encoded_value = raw_value.encode(python_encode_type)
            decoded_value = encoded_value.decode(python_decode_type)
            setattr(dicom, tag, decoded_value)


def copy_dicom_tags(dicom, input_dicom, result_dcm_format='SC'):
    # type1은 원본 꺼를 무조건 넣어 줌!
    type1_elements = [
        'StudyInstanceUID', # General Study
        # 'SpecificCharacterSet',
    ]

    # type2는 원본에 있으면 복사해 주고, 없으면 ''으로 넣어 줌! (즉, 결과에 해당 tag는 항상 생성됨)
    type2_elements = [
        'Modality',
        'PatientName', # Patient
        'PatientID',
        'PatientBirthDate',
        'PatientSex',
        'StudyDate', # General Study
        'StudyTime',
        'ReferringPhysicianName',
        'StudyID',
        'AccessionNumber',
        'StudyDescription', # for SC
    ]
    if result_dcm_format == 'SC' or result_dcm_format == 'MIP':
        type2_elements += [
            'Rows',
            'Columns',
            'PixelSpacing',
        ]
    elif result_dcm_format == 'GSPS':
        type2_elements += [
        ]

    # type3는 원본에 있으면 복사해 주고, 없으면 무시! (즉, 결과에 해당 tag가 없을 수도 있음)
    type3_elements = [
        'PatientAge', # Patient Study
        'PatientSize',
        'PatientWeight',
        'InstitutionName',
        'StudyComments',
        'StationName',
    ]
    if result_dcm_format == 'SC':
        type3_elements += [
            'ViewPosition',
            'Laterality',
            'ImagerPixelSpacing',
            'ImagePositionPatient',
            'ImageOrientationPatient',
            'SpacingBetweenSlice',
        ]
    elif result_dcm_format == 'GSPS':
        type3_elements += [
        ]

    dicom.file_meta = input_dicom.file_meta
    for tag in type1_elements:
        if dicom.__contains__(tag):
            # TODO: ''을 허용하는데, 그러면 안 되는데???
            assert dicom[tag].value in ['', input_dicom[tag].value]
        else:
            assert input_dicom.__contains__(tag)
            dicom.add(input_dicom.data_element(tag))

    tag = 'SpecificCharacterSet'
    if input_dicom.__contains__(tag):
        # 권혁호내과 SpecificCharacterSet이 \ISO 2022 IR 149 일 경우 한글 깨짐 수정!
        # if type(input_dicom.SpecificCharacterSet) == MultiValue:
        #     dicom.SpecificCharacterSet = input_dicom.SpecificCharacterSet[-1]
        # else:
        #     dicom.add(input_dicom.data_element(tag))
        dicom.add(input_dicom.data_element(tag))
    else:
        dicom.SpecificCharacterSet = 'ISO_IR 100'

    for tag in type2_elements:
        if dicom.__contains__(tag):
            assert dicom[tag].value in ['', input_dicom[tag].value]
        else:
            if input_dicom.__contains__(tag):
                dicom.add(input_dicom.data_element(tag))
            else:
                setattr(dicom, tag, '')

    for tag in type3_elements:
        if dicom.__contains__(tag):
            assert dicom[tag].value == input_dicom[tag].value
        else:
            if input_dicom.__contains__(tag):
                dicom.add(input_dicom.data_element(tag))
    copy_dicom_private_tags(dicom, input_dicom)


def copy_dicom_private_tags(dicom, input_dicom):
    private_elements = [
        (0x3355, 0x0010), # Priave Creator
        (0x3355, 0x1001), # AET
        (0x3355, 0x1002), # DEEPAIReqType
        (0x3355, 0x1003), # DEEPAIReqCharset
    ]
    for tag in private_elements:
        if tag in dicom:
            assert dicom[tag].value == input_dicom[tag].value
        else:
            if tag in input_dicom:
                dicom[tag] = input_dicom[tag]

def set_uids(
        dicom,
        input_dicom,
        use_uid=None,
        result_dcm_format='SC',
        sc_idx=1
    ):
    if not use_uid:
        use_uid = uuid.uuid4().int
    try:
        series_idx = int(input_dicom.SeriesNumber)
    except:
        series_idx = 1
    try:
        instance_idx = int(input_dicom.InstanceNumber)
    except:
        instance_idx = 1

    if result_dcm_format == 'SC' or result_dcm_format == 'MIP':
        dicom.SeriesInstanceUID = '2.25.{:d}.{:d}.{:d}'.format(series_idx, sc_idx, use_uid)
        dicom.SOPInstanceUID = '2.25.{:d}.{:d}.{:d}.{:d}'.format(series_idx, sc_idx, instance_idx, use_uid)
        dicom.SeriesNumber = '99991' + str(series_idx)
        dicom.InstanceNumber = str(instance_idx)
    if result_dcm_format == 'RP':
        dicom.SeriesInstanceUID = '2.25.{:d}.2.{:d}'.format(series_idx, use_uid)
        dicom.SOPInstanceUID = '2.25.{:d}.2.{:d}.{:d}'.format(series_idx, instance_idx, use_uid)
        dicom.SeriesNumber = '99992' + str(series_idx)
        dicom.InstanceNumber = str(instance_idx)
    elif result_dcm_format == 'GSPS':
        dicom.SeriesInstanceUID = '2.25.0.0.{:d}'.format(use_uid)
        dicom.SOPInstanceUID = '2.25.0.0.{:d}.{:d}'.format(instance_idx, use_uid) # TODO: [BYK] GSPS 파일 sending 시 중복 방지를 위한 수정
        dicom.SeriesNumber = '999999999'
        dicom.InstanceNumber = '1'


def set_dicom_tags(dicom, product_name, app_name, app_version, result_dcm_format='SC', use_modality=None, intended_uses=None):
    series_tag_values = [
        ('SeriesDescription', product_name),
        ('Manufacturer', 'DEEPNOID'), # General Equipment
    ]
    if result_dcm_format in ['SC', 'RP', 'MIP']:
        if use_modality:
            series_tag_values += [
                ('Modality', use_modality)  # General Series
            ]

        series_tag_values += [
            ('SOPClassUID', '1.2.840.10008.5.1.4.1.1.7'), # SOP Common,
            ('SamplesPerPixel', 3),
            ('PhotometricInterpretation', 'RGB'),
            ('BitsAllocated', 8),
            ('BitsStored', 8),
            ('HighBit', 7),
            ('PlanarConfiguration', 0),
            ('PixelRepresentation', 0), # 0: unsigned integer, 1: 2's complement
            ('RescaleSlope', '1'),
            ('RescaleIntercept', '0'),
            ('RescaleType', 'US'), # Unspecified
            ('ConversionType', 'WSD'), # Workstation, The equipment that placed the modified image into a DICOM SOP instance.
        ]
    elif result_dcm_format == 'GSPS':
        series_tag_values += [
            ('InstanceNumber', 1),
            ('Modality', 'PR'),
            ('SOPClassUID', "1.2.840.10008.5.1.4.1.1.11.1"), # SOP Common,
            ('SoftwareVersions', app_version),
            ('ContentLabel', app_name),
            ('ContentDescription', intended_uses),
            ('ContentCreatorName', 'DEEPNOID'),
            ('PresentationLUTShape', 'IDENTITY'),
        ]

    for tag, value in series_tag_values:
        if dicom.__contains__(tag) and tag != 'Modality':
            assert dicom[tag].value == value
        else:
            setattr(dicom, tag, value)

    dicom.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'


def set_time(dicom, result_dcm_format='SC'):
    creation_date = datetime.now().date().strftime("%Y%m%d")
    creation_time = datetime.now().time().strftime("%H%M%S")
    series_tag_values = [
        ('SeriesDate', creation_date),
        ('SeriesTime', creation_time),
    ]
    if result_dcm_format in ['SC', 'RP', 'MIP']:
        series_tag_values += [
            ('DateOfSecondaryCapture', creation_date), # SC Image Module
            ('TimeOfSecondaryCapture', creation_time),
        ]
    elif result_dcm_format == 'GSPS':
        series_tag_values += [
            ('PresentationCreationDate', creation_date), # Presentation State Identification
            ('PresentationCreationTime', creation_time),
        ]
    for tag, value in series_tag_values:
        setattr(dicom, tag, value)


def set_private_tags(dicom, finding_sequence, app_name, app_version, product_name):
    result_js = {
        'FormatVersion': '1.0',
        'Date': dicom.SeriesDate,
        'Time': dicom.SeriesTime,
        'Vendor': 'DEEPNOID',
        # 'Service': 'DEEP:AI',
        'Service': product_name.upper(),
        'Model': app_name.upper(),
        'Version': app_version
    }
    block = dicom.private_block(0x2021, 'DEEPNOID', create=True)
    block.add_new(0x01, 'LO', "DEEPNOID")
    # block.add_new(0x02, 'LO', 'DEEP:AI')
    block.add_new(0x02, 'LO', product_name.upper())
    block.add_new(0x03, 'LO', app_name.upper())
    block.add_new(0x04, 'LO', app_version)
    # block.add_new(0x05, 'SS', 1)
    # block.add_new(0x06, 'SS', 1)
    block.add_new(0x08, 'UT', json.dumps(result_js))
    block.add_new(0x11, 'SQ', Sequence(finding_sequence))


def get_referenced_image_dataset(input_dicom):
    dataset = Dataset()
    dataset.ReferencedSOPClassUID = input_dicom.SOPClassUID # (0008:1150)
    dataset.ReferencedSOPInstanceUID = input_dicom.SOPInstanceUID # (0008:1155)
    return dataset


def get_referenced_series_dataset(series_instance_uid, referenced_image_sequence):
    dataset = Dataset()
    dataset.SeriesInstanceUID = series_instance_uid
    dataset.ReferencedImageSequence = referenced_image_sequence
    return dataset


def get_graphic_layer_dataset(layer_name, layer_order):
    assert type(layer_name) == str
    assert type(layer_order) == int
    dataset = Dataset()
    dataset.GraphicLayer = layer_name
    dataset.GraphicLayerOrder = layer_order
    return dataset


def get_text_style_dataset(
    style='SOLID',
    color=[255, 255, 255],
    horizontal_align='CENTER',
    vertical_align='CENTER',
    shadow_style='NORMAL',
    shadow_color=[127, 127, 127],
    shadow_offset=[3, 3],
    shadow_opacity=1.0,
    ):

    assert style in ['SOLID', 'DASHED']
    assert shadow_style in ['NORMAL', 'OUTLINED', 'OFF']

    dataset = Dataset()
    dataset.CSSFontName = 'Malgun Gothic'
    dataset.TextColorCIELabValue = rgb2Lab(color)
    dataset.HorizontalAlignment = horizontal_align
    dataset.VerticalAlignment = vertical_align
    dataset.Underlined = 'N'
    dataset.Bold = 'N'
    dataset.Italic = 'N'
    dataset.ShadowStyle = shadow_style
    if shadow_style != 'OFF':
        dataset.ShadowColorCIELabValue = rgb2Lab(shadow_color)
        dataset.ShadowOffsetX = shadow_offset[0]
        dataset.ShadowOffsetY = shadow_offset[1]
        dataset.ShadowOpacity = shadow_opacity
    return dataset


def get_line_style_dataset(
    color,
    thickness,
    style='SOLID',
    opacity=1.0,
    off_color=[0, 0, 0],
    off_opacity=0.0,
    shadow_style='OUTLINED',
    shadow_color=[127, 127, 127],
    shadow_offset=[3, 3],
    shadow_opacity=1.0
    ):

    assert style in ['SOLID', 'DASHED']
    assert shadow_style in ['NORMAL', 'OUTLINED', 'OFF']

    dataset = Dataset()
    dataset.PatternOnColorCIELabValue = rgb2Lab(color)
    dataset.PatternOnOpacity = opacity
    dataset.LineThickness = thickness
    dataset.LineDashingStyle = style
    if style == 'DASHED':
        dataset.PaternOffColorCIELabValue = rgb2Lab(off_color)
        dataset.PatternOffOpacity = off_opacity
        # dataset.LinePattern = 00FFH
    dataset.ShadowStyle = shadow_style
    dataset.ShadowColorCIELabValue = rgb2Lab(shadow_color)
    dataset.ShadowOffsetX = shadow_offset[0]
    dataset.ShadowOffsetY = shadow_offset[1]
    dataset.ShadowOpacity = shadow_opacity
    return dataset


def get_fill_style_dataset(
        color,
        off_color=[127, 127, 127],
        opacity=1.0,
        off_opacity=0.0,
        fill_mode='SOLID'
    ):
    assert fill_mode in ['SOLID', 'STIPPELED']
    dataset = Dataset()
    dataset.PatternOnColorCIELabValue = rgb2Lab(color)
    dataset.PatternOnOpacity = opacity
    dataset.FillMode = fill_mode
    if fill_mode == 'STIPPELED':
        dataset.PatternOffColorCIELabValue = rgb2Lab(off_color)
        dataset.PatternOffOpacity = off_opacity
        # dataset.FillPattern =
    return dataset


def get_graphic_annotation_dataset(
        referenced_image_sequence,
        layer_name,
        text_object_sequence=None,
        graphic_object_sequence=None
    ):
    assert (text_object_sequence is not None) or (graphic_object_sequence is not None)

    dataset = Dataset()
    dataset.ReferencedImageSequence = referenced_image_sequence
    dataset.GraphicLayer = layer_name
    if text_object_sequence is not None:
        dataset.TextObjectSequence = text_object_sequence
    if graphic_object_sequence is not None:
        dataset.GraphicObjectSequence = graphic_object_sequence
    return dataset


def get_text_object_dataset(
        text_data,
        bbox, # (x1, y1, x2, y2)
        text_style,
        boundingbox_annotation_units='PIXEL',
        horizontal_justification='CENTER'
    ):
    dataset = Dataset()
    dataset.BoundingBoxAnnotationUnits = boundingbox_annotation_units
    dataset.BoundingBoxTopLeftHandCorner = [bbox[0], bbox[1]]
    dataset.BoundingBoxBottomRightHandCorner = [bbox[2], bbox[3]]
    dataset.BoundingBoxTextHorizontalJustification = horizontal_justification

    dataset.TextStyleSequence = Sequence([text_style])
    dataset.UnformattedTextValue = text_data
    return dataset


def get_displayed_area_selection(ds, referenced_image_sequence):
    dataset = Dataset()
    dataset.ReferencedImageSequence = referenced_image_sequence
    dataset.PixelOriginInterpretation = 'FRAME'
    dataset.DisplayedAreaTopLeftHandCorner = [1, 1]
    dataset.DisplayedAreaBottomRightHandCorner = [ds.Columns, ds.Rows]
    dataset.PresentationSizeMode = 'SCALE TO FIT'
    dataset.PresentationPixelSpacing = ds.PixelSpacing if 'PixelSpacing' in ds else [1, 1]
    return dataset


def get_graphic_object_dataset(
        graphic_data,
        line_style,
        fill_style=None,
        graphic_annotation_units='PIXEL',
        graphic_dimensions=2,
        graphic_type='POLYLINE',
        graphic_filled='N'
    ):
    assert graphic_annotation_units in ['PIXEL', 'DISPLAY', 'MATRIX']
    assert graphic_type in ['POINT', 'POLYLINE', 'INTERPOLATED', 'CIRCLE', 'ELLIPSE']
    assert graphic_filled in ['Y', 'N']
    dataset = Dataset()
    dataset.GraphicAnnotationUnits = graphic_annotation_units
    dataset.GraphicDimensions = graphic_dimensions
    dataset.NumberOfGraphicPoints = int(len(graphic_data) / graphic_dimensions)
    dataset.GraphicData = graphic_data
    dataset.GraphicType = graphic_type
    dataset.GraphicFilled = graphic_filled
    dataset.LineStyleSequence = Sequence([line_style])
    if fill_style:
        dataset.FillStyleSequence = Sequence([fill_style])
    return dataset


def get_VOI_LUT_dataset(
        input_dicom,
        referenced_image_sequence,
    ):
    dataset = Dataset()
    dataset.ReferencedImageSequence = referenced_image_sequence

    if input_dicom.__contains__('WindowWidth') and input_dicom.__contains__('WindowCenter'):
        dataset.WindowCenter = input_dicom.WindowCenter
        dataset.WindowWidth = input_dicom.WindowWidth
    elif input_dicom.__contains__('VOILUTSequence'):
        dataset.VOILUTSequence = input_dicom.VOILUTSequence
    else:
        # raise KeyError("No 'WindowWidth' nor 'WindowCenter' nor 'VOILUTSequence' in input dicom dataset!")
        pass

    return dataset


def extract_object_by_instance_number(objects, target_instance_number): #########
    for obj in objects:
        if obj['instance_number'] == target_instance_number:
            return obj
    return None


def get_text_object_dcm(text_objects):
    text_object_sequence = []
    for text_object in text_objects:
        text_color = text_object['color'] if 'color' in text_object else [255, 255, 255]
        if 'shadow_color' not in text_object:
            text_style = get_text_style_dataset(
                color=text_color,
                shadow_style='NORMAL',
                shadow_color=[167, 167, 167],
                shadow_offset=[int((text_object['bbox'][3] - text_object['bbox'][1]) / 20)] * 2
            )
        elif text_object['shadow_color'] == 'OFF':
            text_style = get_text_style_dataset(
                color=text_color,
                shadow_style='OFF'
            )
        else:
            text_style = get_text_style_dataset(
                color=text_color,
                shadow_style='NORMAL',
                shadow_color=text_object['shadow_color'],
                shadow_offset=[int((text_object['bbox'][3] - text_object['bbox'][1]) / 20)] * 2
            )
        text_object_sequence.append(
            get_text_object_dataset(
                text_data=text_object["text_data"],
                text_style=text_style,
                bbox=text_object["bbox"]
            )
        )
    return text_object_sequence


def get_graphic_object_dcm(graphic_objects, width, height):
    thickness = float(int(max(width, height) / 1000) + 1)
    graphic_object_sequence = []
    for graphic_object in graphic_objects:
        line_color = graphic_object['color'] if 'color' in graphic_object else [255, 255, 255]
        if 'shadow_color' not in graphic_object:
            line_style = get_line_style_dataset(
                color=line_color,
                thickness=thickness,
                shadow_style='OUTLINED',
                shadow_color=[0, 0, 0],
                shadow_offset=[6, 6]
            )
        elif graphic_object['shadow_color'] == 'OFF':
            line_style = get_line_style_dataset(
                color=line_color,
                thickness=thickness,
                shadow_style='OFF'
            )
        else:
            line_style = get_line_style_dataset(
                color=line_color,
                thickness=thickness,
                shadow_style='OUTLINED',
                shadow_color=graphic_object['shadow_color'],
                shadow_offset=[6, 6]
            )
        graphic_object_sequence.append(
            get_graphic_object_dataset(
                graphic_data=list(graphic_object["graphic_data"].ravel()),
                line_style=line_style
            )
        )
    return graphic_object_sequence


def get_color(pred_json):
    pred = pred_json['pred']
    color = {
        "Benign": [0, 255, 255], #YELLOW
        "Malignant": [0, 0, 255] #RED
    }

    return color[pred]


def get_output_dim(web_json, slide_path, level_dim):
    resize_factor = float(web_json['models']['visualization']['resize_factor']['value'])
    output_dim = (int(level_dim[0][0] * resize_factor), int(level_dim[0][1] * resize_factor))
    return output_dim


def binary2mask(color, binary, output_type):
    canvas = np.ones(binary.shape, np.uint8)
    gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((7, 7), np.uint8)
    dilated_gary = cv2.dilate(gray, kernel, iterations=1)
    if output_type == 'JPEG':
        canvas[gray == 255] = color
        canvas[gray == 0] = [255, 255, 255]
        canvas[gray == 255] = color
        return canvas
    else:
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours


def binary2contour(color, binary, output_type):
    canvas = np.ones(binary.shape, np.uint8)
    canvas[canvas==1] = 255
    gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

    #binary에서 contour 찾기
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if output_type == 'JPEG':
        # contour 그리기
        contour_image = cv2.drawContours(canvas.copy(), contours, -1, color, 3)
        return contour_image #numpy - HxWxC - BGR
    else:
        return contours


def binary2bbox(color, binary, output_type):
    def merge_boxes(boxes):
        merged_boxes = []
    
        for box in boxes:
            # 병합된 바운딩 박스 리스트가 비어있으면 현재 박스 추가
            if not merged_boxes:
                merged_boxes.append(box)
                continue
            
            # 현재 박스와 병합된 바운딩 박스들을 비교하여 겹치는 경우 병합
            merged = False
            for idx, merged_box in enumerate(merged_boxes):
                # 두 바운딩 박스의 겹치는 영역 계산
                x1 = max(box[0], merged_box[0])
                y1 = max(box[1], merged_box[1])
                x2 = min(box[0] + box[2], merged_box[0] + merged_box[2])
                y2 = min(box[1] + box[3], merged_box[1] + merged_box[3])
    
                # 겹치는 영역이 있는 경우
                if x1 < x2 and y1 < y2:
                    # 두 바운딩 박스를 병합한 새로운 바운딩 박스 생성
                    new_box = [min(box[0], merged_box[0]), min(box[1], merged_box[1]),
                                max(box[0] + box[2], merged_box[0] + merged_box[2]) - min(box[0], merged_box[0]),
                                max(box[1] + box[3], merged_box[1] + merged_box[3]) - min(box[1], merged_box[1])]
                    # 병합된 바운딩 박스 리스트에 새로운 바운딩 박스 추가
                    merged_boxes[idx] = new_box
                    merged = True
                    break
                
            # 겹치는 영역이 없는 경우 현재 박스를 병합된 바운딩 박스 리스트에 추가
            if not merged:
                merged_boxes.append(box)
    
        return merged_boxes

    canvas = np.ones(binary.shape, np.uint8)
    canvas[canvas==1] = 255
    
    gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    
    merged_boxes = merge_boxes(bounding_boxes)
    box_count = len(merged_boxes)
    while True:
        merged_boxes = merge_boxes(merged_boxes)
        if box_count == len(merged_boxes):
            break
        else:
            box_count = len(merged_boxes)

    if output_type == 'JPEG':	
        # 병합된 bounding box 그리기
        for box in merged_boxes:
            x, y, w, h = box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
        return canvas
    else:
        return merged_boxes


def output_json(web_json, pred_json, slide_path, save_path, output, vis_type, output_type, binary, level_dim):
    slide_id = slide_path.split('/')[-1]
    opacity = float(web_json['models']['visualization']['opacity']['value'])
    threshold = float(web_json['models']['visualization']['threshold']['value'])
    pred = pred_json['pred']
    prob = float(pred_json['prob'])

    _, b_w, _ = binary.shape
    o_w, _ = level_dim[0]
    mag = o_w / b_w

    result_json = {}
    result_json['type'] = vis_type
    result_json['output_type'] = output_type
    result_json['slide_id'] = slide_id
    if vis_type == 'bounding_box':
        result_json['coord_variable'] = ['x', 'y', 'w', 'h']
        coordinates = []
        for coordinate in output:
            coordinate_dict = {}
            new_coordinate = tuple(int(elem * 16) for elem in coordinate)
            coordinate_dict["coordinate"] = coordinate
            coordinates.append(coordinate_dict)
    else:
        result_json['coord_variable'] = ['x', 'y']
        coordinates = []
        for coordinate in output:
            coordinate_dict = {}
            new_coordinate = coordinate * mag
            new_coordinate = new_coordinate.astype(int)
            coordinate_dict["coordinate"] = new_coordinate.tolist()
            coordinate_dict["coordinate"] = coordinate.tolist()
            coordinates.append(coordinate_dict)

    result_json['coordinates'] = coordinates
    result_json['opacity'] = opacity
    result_json['threshold'] = threshold
    result_json['short_report'] = {'Prediction': pred, 'Probability': prob}
    
    with open(os.path.join(save_path, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=4)


def output_jpeg(web_json, save_path, output, logo_path, slide_path, ops_img, level_dim, app_name, app_version):
    output_dim = get_output_dim(web_json, slide_path, level_dim)
    opacity = float(web_json['models']['visualization']['opacity']['value'])
    
    image = np.array(ops_img.read_region((0, 0), 0, level_dim[0]))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resize_img = cv2.resize(image, output_dim)
    resize_output = cv2.resize(output, output_dim, interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(resize_img, 1-opacity, resize_output, opacity, 0) # numpy
    overlay[resize_output == [255, 255, 255]] = resize_img[resize_output == [255, 255, 255]]

    resize_factor = float(web_json['models']['visualization']['resize_factor']['value'])
    logo = cv2.imread(logo_path)
    o_w, o_h = output_dim[:2]
    # 로고 이미지를 결과이미지 하단 중앙에 겹치기
    logo_arr = np.full(overlay.shape, [255, 255, 255], dtype=np.uint8)
    h, w = logo.shape[:2]
    r_w = int(o_w/7) # 출력 영상 사이즈의 1/7크기
    r_h = int(h * r_w / w)
    resize_logo = cv2.resize(logo, (r_w, r_h))

    start_x = (o_w - r_w) // 2
    start_y = o_h - r_h

    logo_arr[start_y:start_y+r_h, start_x:start_x+r_w, :] = resize_logo
    
    # text 정보
    text = f'{app_name} v{app_version}' # text
    font = cv2.FONT_HERSHEY_SIMPLEX # text 폰트
    font_scale = 1 # text 크기
    color = (43, 43, 43)  # text 색상 (logo_deepgray와 같은 색)
    thickness = 4 # text 두께
    
    # 텍스트 크기 계산
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    # 텍스트 위치 계산
    text_x = start_x + (r_w - text_width) // 2
    text_y = start_y - text_height  # resize_logo 위쪽에 위치
    # 텍스트 넣기
    cv2.putText(logo_arr, text, (text_x, text_y), font, font_scale, color, thickness)

    result = cv2.addWeighted(overlay, 0.5, logo_arr, 0.5, 0)
    result[logo_arr == [255, 255, 255]] = overlay[logo_arr == [255, 255, 255]]

    cv2.imwrite(os.path.join(save_path, 'results.jpg'), result)


def output_xml(web_json, pred_json, slide_path, save_path, output, vis_type, output_type, binary, level_dim):
    slide_id = slide_path.split('/')[-1]
    opacity = float(web_json['models']['visualization']['opacity']['value'])
    threshold = float(web_json['models']['visualization']['threshold']['value'])
    pred = pred_json['pred']
    prob = float(pred_json['prob'])

    _, b_w, _ = binary.shape
    o_w, _ = level_dim[0]
    mag = o_w / b_w
    root = ET.Element('results')
    setting_elem = ET.SubElement(root, 'setting') 
    setting_elem.set('type', vis_type)
    setting_elem.set('output_type', output_type)
    setting_elem.set('slide_id', slide_id)
    setting_elem.set('opacity', str(opacity))
    setting_elem.set('threshold', str(threshold))
    
    result_elem = ET.SubElement(root, 'result')
    result_elem.set('Prediction', pred)
    result_elem.set('Probability', str(prob))

    if vis_type == 'bounding_box':
        for coordinates in output:
            coordinates_elem = ET.SubElement(result_elem, 'coordinates')
            coordinate_elem = ET.SubElement(coordinates_elem, 'coordinate')
            x, y, w, h = coordinates
            x *= mag
            y *= mag
            w *= mag
            h *= mag
            coordinate_elem.set('x', str(x))
            coordinate_elem.set('y', str(y))
            coordinate_elem.set('w', str(w))
            coordinate_elem.set('h', str(h))
    else:
        for i, coordinates in enumerate(output):
            coordinates_elem = ET.SubElement(result_elem, 'coordinates')
            for coordinate in coordinates:
                coordinate_elem = ET.SubElement(coordinates_elem, 'coordinate')
                x, y = coordinate[0]
                x *= mag
                y *= mag
                coordinate_elem.set('x', str(x))
                coordinate_elem.set('y', str(y))
    
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = xml.dom.minidom.parseString(xml_str)
    xml_str_formatted = dom.toprettyxml(indent='\t')

    # XML 파일 저장
    xml_filename = os.path.join(save_path, 'results.xml')
    with open(xml_filename, 'w', encoding='utf-8') as f:
        f.write(xml_str_formatted)


def binary2output(web_json, pred_json, binary_path, vis_type, output_type):
    color = get_color(pred_json)
    binary = cv2.imread(binary_path)
    if vis_type == 'mask':
        output = binary2mask(color, binary, output_type)
    elif vis_type == 'contour':
        output = binary2contour(color, binary, output_type)
    elif vis_type == 'bounding_box':
        output = binary2bbox(color, binary, output_type)
    else:
        raise NotImplementedError
    return output


def create_save_file(web_json, pred_json, slide_path, save_path, binary, logo_path, app_name, app_version):
    import openslide

    vis_type = web_json['models']['visualization']['@type']['selected']
    output_type = web_json['models']['visualization']['@output_type']['selected']
    binary_path = os.path.join(save_path, 'binary.png') 
    binary.save(binary_path)
    output = binary2output(web_json, pred_json, binary_path, vis_type, output_type)
    os.remove(binary_path)

    ops_img = openslide.open_slide(slide_path)
    level_dim = ops_img.level_dimensions

    if output_type == 'JSON':
        output_json(web_json, pred_json, slide_path, save_path, output, vis_type, output_type, binary, level_dim)
    elif output_type == 'JPEG':
        output_jpeg(web_json, save_path, output, logo_path, slide_path, ops_img, level_dim, app_name, app_version)
    elif output_type == 'XML':
        output_xml(web_json, pred_json, slide_path, save_path, output, vis_type, output_type, binary, level_dim)
    else:
        raise NotImplementedError


def create_normal_file(web_json, pred_json, slide_path, save_path, binary, logo_path):    
    import openslide

    def normal_jpeg(web_json, save_path, logo_path, slide_path, ops_img, level_dim):
        output_dim = get_output_dim(web_json, slide_path, level_dim)
        image = np.array(ops_img.read_region((0, 0), 0, level_dim[0]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # resize_factor에 따라 영상 resize
        resize_img = cv2.resize(image, output_dim)

        # 로고 이미지를 결과이미지 하단 중앙에 겹치기
        resize_factor = float(web_json['models']['visualization']['resize_factor']['value'])
        logo = cv2.imread(logo_path)
        resize_logo = cv2.resize(logo, 
                            (int(logo.shape[1]*resize_factor), int(logo.shape[0]*resize_factor)))
        l_h, l_w = resize_logo.shape[:2]
        o_w, o_h = output_dim[:2]
        start_x = (o_w - l_w) // 2
        start_y = o_h - l_h
        result = resize_img.copy()
        result[start_y:start_y+l_h, start_x:start_x+l_w, :] = resize_logo
        result = cv2.addWeighted(result, 0.5, resize_img, 0.5, 0)
        cv2.imwrite(os.path.join(save_path, 'results.jpg'), result)

    def normal_json(web_json, pred_json, slide_path, save_path):
        slide_id = slide_path.split('/')[-1]
        opacity = float(web_json['models']['visualization']['opacity']['value'])
        threshold = float(web_json['models']['visualization']['threshold']['value'])
        pred = pred_json['pred']
        prob = float(pred_json['prob'])

        result_json = {}
        result_json['slide_id'] = slide_id
        result_json['threshold'] = threshold
        result_json['short_report'] = {'Prediction': pred, 'Probability': prob}

        with open(os.path.join(save_path, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=4)

    def normal_xml(web_json, pred_json, slide_path, save_path, binary, level_dim):
        slide_id = slide_path.split('/')[-1]
        opacity = float(web_json['models']['visualization']['opacity']['value'])
        threshold = float(web_json['models']['visualization']['threshold']['value'])
        pred = pred_json['pred']
        prob = float(pred_json['prob'])

        _, b_w, _ = binary.shape
        o_w, _ = level_dim[0]
        mag = o_w / b_w
        root = ET.Element('results')
        setting_elem = ET.SubElement(root, 'setting') 
        setting_elem.set('slide_id', slide_id)
        setting_elem.set('threshold', str(threshold))

        result_elem = ET.SubElement(root, 'result')
        result_elem.set('Prediction', pred)
        result_elem.set('Probability', str(prob))

        xml_str = ET.tostring(root, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_str)
        xml_str_formatted = dom.toprettyxml(indent='\t')

        # XML 파일 저장
        xml_filename = os.path.join(save_path, 'results.xml')
        with open(xml_filename, 'w', encoding='utf-8') as f:
            f.write(xml_str_formatted)

    output_type = web_json['models']['visualization']['@output_type']['selected']

    ops_img = openslide.open_slide(slide_path)
    level_dim = ops_img.level_dimensions

    if output_type == 'JPEG':
        normal_jpeg(web_json, save_path, logo_path, slide_path, ops_img, level_dim)
    elif output_type == 'JSON':
        normal_json(web_json, pred_json, slide_path, save_path)
    elif output_type == 'XML':
        normal_xml(web_json, pred_json, slide_path, save_path, binary, level_dim)
    else:
        raise NotImplementedError


def is_enhanced_dicom(input_path):
    for item in os.listdir(input_path):
        if item.lower().endswith('.dcm'):
            dicom_file_path = os.path.join(input_path, item)
            break

    ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)

    enhanced_uids = {
        "1.2.840.10008.5.1.4.1.1.2.1",      # Enhanced CT
        "1.2.840.10008.5.1.4.1.1.4.1",      # Enhanced MR
        "1.2.840.10008.5.1.4.1.1.130",      # Enhanced PET
        "1.2.840.10008.5.1.4.1.1.12.1.1",   # Enhanced XA
        "1.2.840.10008.5.1.4.1.1.12.2.1"    # Enhanced XRF
    }

    if getattr(ds, "SOPClassUID", None) in enhanced_uids:
        return True

    # Multiframe 기능 그룹 시퀀스 확인
    if hasattr(ds, "PerFrameFunctionalGroupsSequence") or hasattr(ds, "SharedFunctionalGroupsSequence"):
        return True

    return False


# input_path에 있는 Enhanced DICOM 파일을 분리하여 각 프레임을 __split_frames__ 폴더 아래 저장하고, 해당 경로 리턴
def split_enhanced_dicom(input_path: Path):
    dicom_file_path = None
    for item in os.listdir(input_path):
        if item.lower().endswith('.dcm'):
            dicom_file_path = os.path.join(input_path, item)
            break
        
    ds = pydicom.dcmread(dicom_file_path)

    if 'NumberOfFrames' not in ds:
        raise ValueError("Not a multi-frame DICOM file.")

    num_frames = int(ds.NumberOfFrames)
    pixel_array = ds.pixel_array
    per_frame_seq = ds.get('PerFrameFunctionalGroupsSequence', [])
    shared_fg = ds.SharedFunctionalGroupsSequence[0] if 'SharedFunctionalGroupsSequence' in ds else None

    if len(per_frame_seq) != num_frames:
        raise ValueError("Mismatch between NumberOfFrames and PerFrameFunctionalGroupsSequence length")

    split_frames_path: Path = input_path / "__split_frames__"
    split_frames_path.mkdir(parents=True, exist_ok=True)
    _new_ds  = copy.deepcopy(ds)
    for inx in range(num_frames):
        new_ds = _new_ds.copy()
        new_ds.PixelData = pixel_array[inx].tobytes()
        new_ds.NumberOfFrames = 1

        new_ds.SOPInstanceUID = generate_uid()
        new_ds.file_meta.MediaStorageSOPInstanceUID = new_ds.SOPInstanceUID
        new_ds.InstanceNumber = inx + 1

        # Replace sequences
        if shared_fg:
            new_ds.SharedFunctionalGroupsSequence = [shared_fg]
        new_ds.PerFrameFunctionalGroupsSequence = [per_frame_seq[inx]]

        # Remove multi-frame only sequences
        for tag in ['DimensionOrganizationSequence', 'DimensionIndexSequence']:
            if tag in new_ds:
                del new_ds[tag]

        output_path = os.path.join(split_frames_path, f"frame_{inx+1:04d}.dcm")
        new_ds.save_as(output_path)

    return split_frames_path


def merge_dicom_to_enhanced(input_dir):
    dicom_files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {input_dir}")

    dicom_files.sort(key=lambda f: pydicom.dcmread(os.path.join(input_dir, f), stop_before_pixels=True).InstanceNumber)
    template_ds = pydicom.dcmread(os.path.join(input_dir, dicom_files[0]))
    enhanced_ds = template_ds.copy()

    # enhanced_ds.SOPInstanceUID = generate_uid()
    enhanced_ds.file_meta.MediaStorageSOPInstanceUID = enhanced_ds.SOPInstanceUID

    # SOP Class UID 설정
    modality = enhanced_ds.Modality
    sop_class_map = {
        'MR': '1.2.840.10008.5.1.4.1.1.4.1',  # Enhanced MR
        'CT': '1.2.840.10008.5.1.4.1.1.2.1',  # Enhanced CT
        'SC': '1.2.840.10008.5.1.4.1.1.7.1'   # Multi-frame SC
    }
    enhanced_ds.SOPClassUID = sop_class_map.get(modality, sop_class_map['SC'])
    enhanced_ds.file_meta.MediaStorageSOPClassUID = enhanced_ds.SOPClassUID

    pixel_arrays = []
    per_frame_functional_groups = []
    ref_sop_seq_items = []

    for filename in dicom_files:
        file_path = os.path.join(input_dir, filename)
        ds = pydicom.dcmread(file_path)
        pixel_arrays.append(ds.pixel_array)

        # 기능 그룹 시퀀스 구성
        frame_fg = pydicom.Dataset()

        if hasattr(ds, 'ImagePositionPatient'):
            pos = pydicom.Dataset()
            pos.ImagePositionPatient = ds.ImagePositionPatient
            frame_fg.PlanePositionSequence = [pos]

        if hasattr(ds, 'ImageOrientationPatient'):
            ori = pydicom.Dataset()
            ori.ImageOrientationPatient = ds.ImageOrientationPatient
            frame_fg.PlaneOrientationSequence = [ori]

        if hasattr(ds, 'PixelSpacing'):
            pm = pydicom.Dataset()
            pm.PixelSpacing = ds.PixelSpacing
            if hasattr(ds, 'SliceThickness'):
                pm.SliceThickness = ds.SliceThickness
            frame_fg.PixelMeasuresSequence = [pm]

        per_frame_functional_groups.append(frame_fg)

        # 참조 SOP 아이템 추가
        ref_item = pydicom.Dataset()
        ref_item.ReferencedSOPClassUID = ds.SOPClassUID
        ref_item.ReferencedSOPInstanceUID = ds.SOPInstanceUID
        ref_sop_seq_items.append(ref_item)

        os.remove(file_path)

    # 픽셀 병합
    combined_pixel_array = np.stack(pixel_arrays, axis=0)
    enhanced_ds.PixelData = combined_pixel_array.tobytes()
    enhanced_ds.NumberOfFrames = len(dicom_files)

    # 공유/프레임 기능 그룹 설정
    enhanced_ds.SharedFunctionalGroupsSequence = [pydicom.Dataset()]
    enhanced_ds.PerFrameFunctionalGroupsSequence = per_frame_functional_groups

    # Dimension 정보 설정
    dim_uid = generate_uid()
    enhanced_ds.DimensionOrganizationSequence = [pydicom.Dataset(DimensionOrganizationUID=dim_uid)]

    dim_index = pydicom.Dataset()
    dim_index.DimensionOrganizationUID = dim_uid
    dim_index.DimensionIndexPointer = (0x5200, 0x9230)  # PerFrameFunctionalGroupsSequence
    dim_index.FunctionalGroupPointer = (0x0020, 0x9113)  # PlanePositionSequence
    dim_index.DimensionDescriptionLabel = "StackID"
    enhanced_ds.DimensionIndexSequence = [dim_index]

    # ReferencedImageEvidenceSequence 설정
    ref_series = pydicom.Dataset()
    ref_series.ReferencedSOPSequence = ref_sop_seq_items
    ref_series.SeriesInstanceUID = template_ds.SeriesInstanceUID

    ref_study_item = pydicom.Dataset()
    ref_study_item.ReferencedSeriesSequence = [ref_series]
    ref_study_item.StudyInstanceUID = template_ds.StudyInstanceUID

    enhanced_ds.ReferencedImageEvidenceSequence = [ref_study_item]
    
    # Enhanced DICOM 파일 저장
    enhanced_ds.save_as(str(input_dir / (enhanced_ds.SOPInstanceUID + '.dcm')))
    # print(f"Enhanced DICOM with {len(dicom_files)} frames saved to '{output_path}'", flush=True)


# Additional utility functions for HER2 processing
def load_dicom_image(dicom_path):
    """
    Load a DICOM image and convert to RGB array
    
    Args:
        dicom_path: Path to DICOM file
        
    Returns:
        RGB numpy array
    """
    ds = pydicom.dcmread(dicom_path)
    image_array = read_dcm(ds)
    
    # Ensure RGB format
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    
    return image_array.astype(np.uint8)


def load_wsi_image(wsi_path):
    """
    Load a whole slide image
    
    Args:
        wsi_path: Path to WSI file
        
    Returns:
        WSI object (OpenSlide or PIL Image)
    """
    try:
        import openslide
        wsi = openslide.open_slide(wsi_path)
        return wsi
    except ImportError:
        # Fallback to PIL for smaller images
        from PIL import Image
        return Image.open(wsi_path)


def extract_wsi_tiles(wsi, tile_size, overlap):
    """
    Extract tiles from a WSI
    
    Args:
        wsi: WSI object
        tile_size: Size of each tile
        overlap: Overlap between tiles
        
    Yields:
        Tuple of (tile, coordinates)
    """
    try:
        # Check if it's an OpenSlide object
        if hasattr(wsi, 'dimensions'):
            width, height = wsi.dimensions
            step = tile_size - overlap
            
            for y in range(0, height - tile_size + 1, step):
                for x in range(0, width - tile_size + 1, step):
                    # Read region
                    tile = wsi.read_region((x, y), 0, (tile_size, tile_size))
                    tile = np.array(tile)[:, :, :3]  # Remove alpha channel
                    
                    # Check if tile contains tissue (simple check)
                    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
                    if np.mean(gray) < 240:  # Not all white
                        yield tile, (x, y)
        else:
            # PIL Image fallback
            width, height = wsi.size
            step = tile_size - overlap
            
            for y in range(0, height - tile_size + 1, step):
                for x in range(0, width - tile_size + 1, step):
                    tile = wsi.crop((x, y, x + tile_size, y + tile_size))
                    tile = np.array(tile)
                    
                    # Check if tile contains tissue
                    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
                    if np.mean(gray) < 240:  # Not all white
                        yield tile, (x, y)
    except Exception as e:
        print(f"Error extracting tiles: {e}")
        # Return empty if error
        return
