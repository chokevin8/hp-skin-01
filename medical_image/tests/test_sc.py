import os
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import medical_image
from medical_image.utils import read_dcm
from pathlib import Path


def test_sc_2D(ds, result_json):
    image = Image.fromarray(read_dcm(ds).astype(np.uint8))
    img_editable = ImageDraw.Draw(image)
    ## GraphicObject
    contour_image_array = np.array(image)
    for graphic_annotation in result_json["GraphicAnnotationSequence"]:
        heatmap_str = graphic_annotation['Heatmap']
        if heatmap_str != None:
            heatmap = np.array(Image.open(BytesIO(base64.b64decode(heatmap_str))))
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGRA2BGR)
            contour_image_array = cv2.addWeighted(contour_image_array, 1, heatmap, 0.5, 0)

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
                font_scale = int(result_json['Height'] / 70)

            font = ImageFont.truetype(os.path.join(medical_image.__path__[0], f'fonts/NotoSansCJKkr-{font_type}.otf'), font_scale)
            text_color = text_object['color'] if 'color' in text_object else (255, 255, 255)
            msg = text_object['text_data']
            x, y, w, h = text_object['bbox']
            if ('shadow_color' not in text_object) or (text_object['shadow_color'] != 'OFF'):
                # mw, mh, _, _ = font.getbbox(msg)
                mw, mh = font.getsize(msg)
                shadow = int(mh / 20)
                shadow_color = text_object['shadow_color'] if 'shadow_color' in text_object else (167, 167, 167)
                img_editable.text((x + shadow, y + shadow), msg, shadow_color, font)
            img_editable.text((x, y), msg, text_color, font)

    image.save('test_sc_2D.png')


def test_sc_3D(ds_dict, result_json):
    instance_numbers = sorted([key for key in ds_dict.keys() if type(key) == int])

    object_numbers = {}
    for i, graphic_annotation in enumerate(result_json["GraphicAnnotationSequence"]):
        objects = graphic_annotation['Objects']
        _ins = [obj['instance_number'] for obj in objects]
        for _in in _ins:
            if _in in object_numbers:
                object_numbers[_in][i] = _ins.index(_in)
            else:
                object_numbers[_in] = {i: _ins.index(_in)}

    result_path = Path('./test_sc_3D')
    result_path.mkdir(exist_ok=True)
    for j, instance_number in enumerate(instance_numbers):
        ds = ds_dict[instance_number]

        objects = []
        if instance_number in object_numbers:
            for i in object_numbers[instance_number]:
                objects.append(result_json["GraphicAnnotationSequence"][i]["Objects"][object_numbers[instance_number][i]])

        if len(objects):
            height = objects[0]['Height']
            width = objects[0]['Width']
        else:
            height = int(ds.Rows)
            width = int(ds.Columns)

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

        image.save(str(result_path / '{:03d}.png'.format(j + 1)))