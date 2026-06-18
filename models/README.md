# Model Weights

The trained weights are too large for GitHub.

## Download

| File | Description | Link |
|------|-------------|------|
| `ppe_yolo11n_final.pt` | PyTorch weights (best epoch) | Kaggle Output Tab |
| `ppe_yolo11n_final.onnx` | ONNX export (deployment) | Kaggle Output Tab |

After downloading from Kaggle, place them in this `models/` folder.

## Usage

```python
from src.detector import PPEViolationDetector

detector = PPEViolationDetector(
    model_path='models/ppe_yolo11n_final.pt',
    class_names=['hardhat','no-hardhat','safety-vest','no-safety-vest',
                 'mask','no-mask','gloves','person'],
    conf_threshold=0.35
)
```
