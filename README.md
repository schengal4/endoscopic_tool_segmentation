# MONAI Endoscopic Tool Segmentation API

A web service API for real-time segmentation of surgical tools in endoscopic images, built with FastAPI and powered by MONAI's advanced deep learning capabilities.

![Endoscopic Tool Segmentation Example](https://via.placeholder.com/800x400)

## Overview

This API provides a simple interface to a powerful AI model for identifying and segmenting surgical tools in endoscopic images. The model employs a flexible UNet architecture with an EfficientNet-B2 backbone, achieving a mean IoU score of 0.86.

## Features

- **Web Interface**: Easy-to-use web UI for uploading and processing images
- **REST API**: Documented endpoints for programmatic integration
- **High Performance**: Optimized with TensorRT acceleration for faster inference
- **Flexible Deployment**: Configurable for various environments and hardware
- **Comprehensive Results**: Returns original, segmentation mask, and blended visualizations

## Technical Specifications

### Model Architecture
- EfficientNet-B2 encoder backbone
- UNet decoder architecture
- Binary segmentation (tools vs. background)
- Input resolution: 736 x 480 x 3

### Performance
- Mean IoU score: 0.86
- TensorRT acceleration with up to 2.3x speedup in FP16 precision
- Benchmarked on NVIDIA A100 80G GPU

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface for the tool segmentation service |
| GET | `/example` | Processes example images to demonstrate the API |
| POST | `/segment` | Upload and process endoscopic images |
| GET | `/download/{session_id}/{path}` | Download specific result files |

### Example API Usage

```bash
# Process images via curl
curl -X POST "http://localhost:5000/segment" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.9+
- MONAI

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/monai-endoscopic-segmentation.git
   cd monai-endoscopic-segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model:
   ```bash
   # Create directories
   mkdir -p models configs
   
   # Download model (for example purposes)
   # Replace with the actual download command
   python download_model.py
   ```

4. Start the server:
   ```bash
   python app.py
   ```

5. Access the web interface at `http://localhost:5000`

## Development and Customization

### File Structure
```
.
├── app.py              # Main FastAPI application
├── configs/            # Model configuration files
├── models/             # Pre-trained model weights
├── real/               # Example images for demonstration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

### Adding Custom Models

To use a different model:

1. Place your model checkpoint in the `models/` directory
2. Update the configuration in `configs/inference.json`
3. Modify the preprocessing/postprocessing steps in `app.py` if necessary

## Acknowledgements

This project is built upon the work from the MONAI Consortium. The original model architecture and training methodology were developed by the researchers at Activ Surgical, as detailed in the [original repository](https://github.com/project-monai/model-zoo/tree/dev/models/endoscopic_tool_segmentation).

The model uses a UNet architecture with an EfficientNet-B2 backbone. For more information on these architectures, see:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf) (Tan & Le, 2019)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf) (Ronneberger et al., 2015)

## License

This project is licensed under the Apache License, Version 2.0 - see the original [LICENSE](http://www.apache.org/licenses/LICENSE-2.0) for details.

## Citing This Work

If you use this tool in your research or project, please cite the original MONAI model:

```
@misc{monai_endoscopic_tool_segmentation,
  author = {MONAI Consortium},
  title = {Endoscopic Tool Segmentation},
  year = {2023},
  publisher = {MONAI Model Zoo},
  howpublished = {\url{https://github.com/project-monai/model-zoo/tree/dev/models/endoscopic_tool_segmentation}}
}
```