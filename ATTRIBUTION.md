# Attribution and licences

## Datasets
* **Printed Circuit Board dataset (Roboflow 100), version 4 "release-filtered"**,
  https://universe.roboflow.com/roboflow-100/printed-circuit-board/dataset/4.
  Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
  Originally created by T. Pankaj, Aniket Dhanotia, Tejasvi Singh, Garvita Vijay et al.
  (https://universe.roboflow.com/new-workspace-rzrja/pcb-2.0). Part of RF100 (https://rf100.org).
  **Changes made:** labels merged from 23 to 7 classes, non-component labels removed, images resized,
  augmented duplicates removed, and a synthetic asset tag added to the 50 demo photos.
* **pcb-components detection, version 1** (fedoration),
  https://universe.roboflow.com/fedoration/pcb-components-detection/dataset/1. CC BY 4.0.
  Examined but not used (its labels are unnamed numbers).

## Models and libraries
* YOLOv8 by Ultralytics (AGPL-3.0), used for training. The exported detector is used for this academic project.
* Vosk speech recognition and the `vosk-model-small-en-us-0.15` model (Apache 2.0), https://alphacephei.com/vosk.
* Google ML Kit Text Recognition (bundled model), under the ML Kit terms of service.
* ONNX Runtime (MIT), scikit-learn (BSD-3), Tesseract OCR (Apache 2.0), Whisper / faster-whisper (MIT), Streamlit (Apache 2.0).
