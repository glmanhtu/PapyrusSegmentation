### Pre-requirements
```bash
pip install -r requirements.txt
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
cd ..
```

Download pretrained models:
```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
```bash
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth
```

### Segmentation
Run `papyrus_segmentation.py` script for segmentation and `papyrus_detection.py` for detection.

```bash
python3 papyrus_segmentation.py --dataset_path /path/to/dataset --output_path /path/to/output
```

```bash
python3 papyrus_detection.py --dataset_path /path/to/dataset --output_path /path/to/output
```