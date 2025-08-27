# MarkerAI-Skin

Epidermis segmentation software for skin histopathology whole slide image analysis.

## Installation

```bash
# Create conda environment
conda create -n markerai-skin python=3.9
conda activate markerai-skin

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line
```bash
# Process a single WSI
python main.py path/to/wsi.jpg --output-format JSON

# Process with specific output directory
python main.py path/to/wsi.tif --output-dir ./results --output-format JSON

# Process with custom configuration
python main.py path/to/image.png --config configuration/deepai_epidermis.json --output-format JSON
```

## Core Components

### EpidermisPipeline (main.py)
Main pipeline class that orchestrates the entire segmentation workflow:
- **Inherits from DicomModuleHisto**: Provides medical imaging I/O compliance and future DICOM support
- **Security integration**: Uses security_helper for operation modes (DEV/SECURED)
- **WSI processing**: Handles large whole slide images by patch-based processing
- **Tissue segmentation**: CLAM-inspired tissue detection for efficient processing

### EpidermisClassifier (models/epidermis_classifier.py)
Core segmentation algorithm using DeepLabV3+ with EfficientNet-B3 encoder:
- **Deep learning model**: Pre-trained on skin histopathology datasets
- **Patch-based inference**: Processes 384×384 patches
- **Multi-scale support**: Handles 10x and 20x magnifications

### EpidermisPostProcessor (postprocessing_epidermis.py)
Handles output generation and morphological analysis:
- **Contour extraction**: Converts segmentation masks to coordinate lists
- **Morphological metrics**: Calculates area and perimeter in mm²/mm
- **JSON generation**: Creates structured output with contours and measurements

### DicomModuleHisto (medical_image/generate_dcm.py)
Base class for medical imaging modules:
- **Standardized I/O**: Provides consistent interface for medical images
- **DICOM support**: Ready for DICOM input integration
- **WSI handling**: Supports various WSI formats through medical_image.utils
- **Future extensibility**: New methods added here automatically available to EpidermisPipeline

## Output Format

The pipeline generates a JSON file containing:
- `type`: 'contour'
- `output_type`: 'JSON'
- `slide_id`: WSI identifier
- `short_report`: Morphological measurements (area_mm2, perimeter_mm)
- `coordinates`: Contour coordinates for epidermis regions

## Segmentation Target
- **Epidermis**: The outermost layer of skin tissue
- **Binary segmentation**: Epidermis (1) vs. Other tissue (0)
- **Multi-dataset trained**: Robust to staining variations

## Configuration

Main configuration in `configuration/deepai_epidermis.json`:
- Model settings (encoder, checkpoint path)
- Tissue segmentation parameters
- Patch processing settings
- Visualization options
- Output directory settings

## Future Integration

Since EpidermisPipeline inherits from DicomModuleHisto, future DICOM support can be seamlessly integrated. When DicomModuleHisto adds DICOM processing methods, they will automatically be available to the pipeline through inheritance, requiring minimal code changes.

## Requirements
- Python 3.9+
- CUDA-capable GPU (optional, for acceleration)
- See `requirements.txt` for complete dependencies