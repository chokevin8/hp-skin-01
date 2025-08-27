import numpy as np


def test_text_objects_2D(text_objects):
    for text_object in text_objects:
        assert type(text_object) == dict, type(text_object)
        assert type(text_object['text_data']) == str, text_object['text_data']
        assert type(text_object['bbox']) == list, type(text_object['bbox'])
        assert len(text_object['bbox']) == 4, text_object['bbox']
        if 'color' in text_object:
            assert type(text_object['color']) == tuple, type(text_object['color'])
            assert len(text_object['color']) == 3, text_object['color']
            assert text_object['color'][0] >= 0, text_object['color'][0]
            assert text_object['color'][0] <= 255, text_object['color'][0]
            assert text_object['color'][1] >= 0, text_object['color'][1]
            assert text_object['color'][1] <= 255, text_object['color'][1]
            assert text_object['color'][2] >= 0, text_object['color'][2]
            assert text_object['color'][2] <= 255, text_object['color'][2]
        if 'shadow_color' in text_object:
            assert (type(text_object['shadow_color']) == tuple) or (text_object['shadow_color'] == 'OFF'), text_object['shadow_color']
            if type(text_object['shadow_color']) == tuple:
                assert len(text_object['shadow_color']) == 3, text_object['shadow_color']
                assert text_object['shadow_color'][0] >= 0, text_object['shadow_color'][0]
                assert text_object['shadow_color'][0] <= 255, text_object['shadow_color'][0]
                assert text_object['shadow_color'][1] >= 0, text_object['shadow_color'][1]
                assert text_object['shadow_color'][1] <= 255, text_object['shadow_color'][1]
                assert text_object['shadow_color'][2] >= 0, text_object['shadow_color'][2]
                assert text_object['shadow_color'][2] <= 255, text_object['shadow_color'][2]
        if 'font_type' in text_object:
            assert text_object['font_type'] in ['Thin', 'DemiLight', 'Medium', 'Bold'], text_object['font_type']
        if 'font_scale' in text_object:
            assert type(text_object['font_scale']) == int, type(text_object['font_scale'])


def test_graphic_objects_2D(graphic_objects):
    for graphic_object in graphic_objects:
        assert type(graphic_object) == dict, type(graphic_object)
        assert type(graphic_object['graphic_data']) == np.ndarray, type(graphic_object['graphic_data'])
        assert graphic_object['graphic_data'].ndim == 2, graphic_object['graphic_data'].shape
        for xy in graphic_object['graphic_data']:
            assert len(xy) == 2, xy
        if 'color' in graphic_object:
            assert type(graphic_object['color']) == tuple, graphic_object['color']
            assert len(graphic_object['color']) == 3, graphic_object['color']
            assert graphic_object['color'][0] >= 0, graphic_object['color'][0]
            assert graphic_object['color'][0] <= 255, graphic_object['color'][0]
            assert graphic_object['color'][1] >= 0, graphic_object['color'][1]
            assert graphic_object['color'][1] <= 255, graphic_object['color'][1]
            assert graphic_object['color'][2] >= 0, graphic_object['color'][2]
            assert graphic_object['color'][2] <= 255, graphic_object['color'][2]
        if 'thickness' in graphic_object:
            assert type(graphic_object['thickness']) == int, type(graphic_object['thickness'])
            if (graphic_object['thickness'] == -1) and ('fill_opacity' in graphic_object):
                assert type(graphic_object['fill_opacity']) in [float, np.float32, np.float64], type(graphic_object['fill_opacity'])
                assert graphic_object['fill_opacity'] >= 0., graphic_object['fill_opacity']
                assert graphic_object['fill_opacity'] <= 1., graphic_object['fill_opacity']
        if 'shadow_color' in graphic_object:
            assert (type(graphic_object['shadow_color']) == tuple) or (graphic_object['shadow_color'] == 'OFF'), graphic_object['shadow_color']
            if type(graphic_object['shadow_color']) == tuple:
                assert len(graphic_object['shadow_color']) == 3, graphic_object['shadow_color'] 
                assert graphic_object['shadow_color'][0] >= 0, graphic_object['shadow_color'][0]
                assert graphic_object['shadow_color'][0] <= 255, graphic_object['shadow_color'][0]
                assert graphic_object['shadow_color'][1] >= 0, graphic_object['shadow_color'][1]
                assert graphic_object['shadow_color'][1] <= 255, graphic_object['shadow_color'][1]
                assert graphic_object['shadow_color'][2] >= 0, graphic_object['shadow_color'][2]
                assert graphic_object['shadow_color'][2] <= 255, graphic_object['shadow_color'][2]


def test_type_2D(result_json):
    assert type(result_json) == dict, type(result_json)
    for key in result_json:
        print("type(result_json['{}']):".format(key), type(result_json[key]))
    assert type(result_json['Height']) == int, result_json['Height']
    assert type(result_json['Width']) == int, result_json['Width']
    assert type(result_json['GraphicAnnotationSequence']) == list, type(result_json['GraphicAnnotationSequence'])
    assert (type(result_json['SCArray']) == np.ndarray) or (result_json['SCArray'] == None), type(result_json['SCArray'])
    assert (type(result_json['ReportArray']) == np.ndarray) or (result_json['ReportArray'] == None), type(result_json['ReportArray'])
    assert (type(result_json['ImageComments']) == str) or (result_json['ImageComments'] == None), result_json['ImageComments']
    assert type(result_json['Skipping']) == bool, result_json['Skipping']

    for i, graphic_annotation in enumerate(result_json['GraphicAnnotationSequence']):
        assert type(graphic_annotation) == dict, type(graphic_annotation)
        for key in graphic_annotation:
            print("type(result_json['GraphicAnnotationSequence'][{}]['{}']):".format(i, key), type(graphic_annotation[key]))
        assert type(graphic_annotation['GraphicLayer']) == str, graphic_annotation['GraphicLayer']
        assert type(graphic_annotation['TextObjectSequence']) == list, type(graphic_annotation['TextObjectSequence'])
        assert type(graphic_annotation['GraphicObjectSequence']) == list, type(graphic_annotation['GraphicObjectSequence'])
        assert type(graphic_annotation['Probability']) in [float, np.float32, np.float64], type(graphic_annotation['Probability'])
        assert graphic_annotation['Probability'] >= 0., graphic_annotation['Probability']
        assert graphic_annotation['Probability'] <= 1., graphic_annotation['Probability']
        assert (type(graphic_annotation['Heatmap']) == str) or (graphic_annotation['Heatmap'] == None), graphic_annotation['Heatmap']

        test_text_objects_2D(graphic_annotation['TextObjectSequence'])
        test_graphic_objects_2D(graphic_annotation['GraphicObjectSequence'])


def test_type_3D(result_json):
    pass

if __name__ == '__main__':
    pass
