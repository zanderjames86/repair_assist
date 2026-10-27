# PCB Repair Assistant

**Live web app:** https://pcb-repair-assistant.streamlit.app ·
**Android APK:** [Releases → latest](https://github.com/zanderjames86/repair_assist/releases/tag/latest)

Hands-free job cards for the electronics repair bench. A technician photographs a faulty board,
speaks a short note about the fault, and the assistant reads the board's asset tag, finds the
components on the board, diagnoses the fault and recommends whether the board is worth repairing.

It ships as two apps built on the same models:

| | **Android app** (main tool) | **Web app** (Streamlit) |
|---|---|---|
| Runs | Fully offline on a phone or tablet: no internet permission at all | In a browser, hosted free on Streamlit Community Cloud |
| OCR | Google ML Kit text recognition (bundled model) | Tesseract |
| Speech-to-text | Vosk small English model (bundled) | Whisper `base.en` (faster-whisper) |
| Part detector | YOLOv8n ONNX via ONNX Runtime | same ONNX model |
| ML models | Exported weights, run in Java; unit-tested to match scikit-learn to 1e-6 | scikit-learn |
| Install | Copy `PCB-Repair-Assistant.apk` to the tablet over USB | Open the link |

## AI and ML techniques

| Type | Technique | What it does |
|---|---|---|
| **AI** (deep learning, pretrained) | **OCR** | Reads board family, revision and serial number from the asset tag |
| **AI** (deep learning, pretrained) | **Speech-to-text** | Transcribes the technician's voice note |
| **AI** (deep learning, **fine-tuned by us**) | **Object detection**: YOLOv8n | Finds and classifies components: IC, capacitor, resistor, connector, transistor/diode, inductor, switch |
| **ML** (classical, **trained by us**) | **Text classification**: TF-IDF + Logistic Regression | Predicts the fault category from the note and shows which words drove the decision |
| **ML** (classical, **trained by us**) | **Ensemble learning**: Random Forest classifier + regressor | Predicts Repairable vs BER (beyond economic repair) and estimated repair hours |

## Workflow on the bench

1. **Photograph the board** (camera, gallery or one of 50 demo boards). OCR finds the yellow asset
   tag, crops and cleans it, and reads it. The serial number is looked up in the board registry.
2. **Components:** the detector draws a box around every part and counts them by type.
3. **Voice note:** *"Green deposits on the pins, salt residue across the lower edge, moisture ingress."*
4. **Fault diagnosis (ML 1):** probability for each of 6 fault categories, plus the words that drove it.
5. **Repair recommendation (ML 2):** board type, age, previous repairs and the predicted fault give
   BER risk and estimated hours.
6. **Save** the job card. The log is exported as CSV (to the tablet's storage or a USB drive).

## Results

All scores are on held-out data.

**Fault classifier** (120 held-out notes, 5-fold cross-validation for model selection)

| Model | CV F1 | Test accuracy |
|---|---|---|
| Naive Bayes | 0.969 | 0.950 |
| **Logistic Regression** (chosen: probabilities + explanations) | 0.977 | 0.950 |
| Linear SVM | 0.977 | 0.942 |

**Repair outcome predictor (Random Forest):** BER ROC AUC 0.82, accuracy 75% (always guessing "Repairable"
gives 72%). Repair-hours mean absolute error 0.43 h, against 1.24 h for guessing the average. Board age and
fault type drive scrap decisions.

**Part detector** (YOLOv8n, COCO-pretrained, fine-tuned for 138 epochs / 4.8 h on a 4-core CPU runner;
scored on 22 test boards never used for training or model selection):

| | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| All 7 part types | **0.349** | 0.184 | 0.418 | 0.371 |

| IC | Switch | Connector | Resistor | Capacitor | Transistor/Diode | Inductor |
|---|---|---|---|---|---|---|
| 0.63 | 0.59 | 0.54 | 0.26 | 0.24 | 0.17 | 0.02 |

Large, distinctive parts (ICs, switches, connectors) are detected well. Tiny SMD passives on dense boards are
only a few pixels wide at 1024 px, and inductors have just 105 training examples, so those classes are weak.
The best validation mAP@50 was 0.40 (epoch 118). One detection takes about 0.2 s on a laptop CPU.
A first 4-hour run scored 0.286; most of that gap was an evaluation artefact. NMS hit its time limit and the
300-box cap on dense boards, so detections were dropped. Both limits were raised for the final run.

**OCR on the 50 demo photos:**

| Pipeline | Serials read correctly |
|---|---|
| Web app: Tesseract on the whole photo | 15 / 50 (30%) |
| Web app: find tag → crop → contrast + Otsu → Tesseract | **49 / 50 (98%)** |
| Android: ML Kit on the device (emulator test) | **47 / 50 (94%)** |

The Tesseract miss is a yellow tag on a yellow motherboard.

**Speech-to-text** (4 synthesised voice notes, 23 key words; tests in CI):

| Engine | Key words recognised | Faults diagnosed correctly from the transcript |
|---|---|---|
| Whisper base.en (web app) | 18 / 23 | 4 / 4 |
| Vosk small-en, free vocabulary (phone) | 7 / 23 | not reached |
| Vosk small-en + domain word list (phone, shipped) | **21 / 23** | **4 / 4** |

The phone constrains Vosk to a 279-word repair vocabulary: every word the fault classifier knows plus
an electronics lexicon. Accuracy on the robotic test voice went from 30% to 91%. The word list is built from
the training notes, never from the test sentences.

## Data

* **Repair records:** 600 synthetic job records (`generate_data.py`, fixed seed). Notes are built from
  phrase banks for each fault type, and 25% of them mix in a symptom from another fault. No real facility data is used.
* **PCB photos:** [Roboflow 100 "printed circuit board" v4](https://universe.roboflow.com/roboflow-100/printed-circuit-board/dataset/4),
  CC BY 4.0. The 23 original labels are merged into 7 part types, and non-parts (pads, test points,
  unknown) are dropped. Roboflow's split already keeps each board in a single split, which avoids leakage.
  137 boards are used for training, 40 for validation and 22 for testing.
* **Demo boards:** 50 of the 62 validation and test boards (the detector never trained on them), each with a
  synthetic yellow asset tag and a synthetic history in `data/board_registry.csv`.
* We also examined [pcb-components detection v1](https://universe.roboflow.com/fedoration/pcb-components-detection/dataset/1)
  (CC BY 4.0). Its 68 labels are bare numbers ("1" to "67", "object") with no published meaning, so they
  can't be mapped to part types, and it is not used for training.

See [ATTRIBUTION.md](ATTRIBUTION.md) for credits and licences.

## Testing

Every push runs on GitHub Actions:

* **Python** (`tests/`, pytest):
  * data integrity and reproducibility;
  * model quality thresholds and sensible behaviour (e.g. BER risk rises with age);
  * retraining reproduces the metrics;
  * OCR accuracy on all 50 demo photos;
  * the exported Android models match scikit-learn;
  * detector decoding and quality;
  * a headless Streamlit run of the full bench flow.
* **Android JVM unit tests:**
  * on-device fault classifier and Random Forests match scikit-learn on 600 cases to 1e-6;
  * YOLO decoding matches the Python decoder on 349 detections;
  * OCR label parsing, including common OCR errors.
* **Android emulator end-to-end test:**
  * the app has no INTERNET permission;
  * ML Kit reads at least 90% of the 50 demo tags (94% measured);
  * the 600 parity cases pass again on Android's own runtime;
  * offline speech-to-text: synthesised WAV notes → Vosk → fault classifier;
  * the detector finds parts;
  * a full job card is created through the UI and appears in the log.
* **Build check:** the release APK is inspected to confirm it has no network permission.

### Bugs the automated tests caught

* Android's regex engine rejects a Java flag used in the TF-IDF tokenizer, which would have crashed the app on
  any real device. Caught by the emulator test.
* A PyAV release removed an argument faster-whisper relies on, which broke web voice notes. Audio is now decoded with ffmpeg.
* A process-wide thread limit set for Tesseract made Whisper produce garbage transcripts. It's now scoped
  to Tesseract calls, with a regression test.
* Running Tesseract with several threads per image while reading photos in parallel caused CI timeouts.

## Install the Android app (air-gapped tablet)

1. Download `PCB-Repair-Assistant.apk` from the repo's **Releases** page (on any internet-connected PC).
2. Copy it to the tablet by USB cable or a USB drive.
3. On the tablet, open it from the Files app and allow "install unknown apps" for Files.
4. The app needs no network. Grant microphone permission the first time you record a voice note.

Requires Android 8.0 or later. The APK is about 127 MB because every model is bundled (Vosk speech ~40 MB, detector 13 MB, ML Kit OCR).

## Run the web app

```bash
sudo apt install tesseract-ocr        # Windows: UB Mannheim installer
pip install -r requirements.txt
streamlit run app.py
```

To deploy on Streamlit Community Cloud, connect this repo, set main file `app.py` and Python 3.12.
`packages.txt` installs Tesseract.

## Rebuild everything

```bash
python generate_data.py                     # synthetic repair records
python train_models.py                      # ML models + metrics (about 5 s)
python export_android.py                    # JSON weights + parity tests for the phone
python detector/prepare_dataset.py <rf100>  # detector dataset from the Roboflow zip
python make_demo_images.py <rf100>          # 50 demo boards + registry
python make_detector_fixture.py             # shared YOLO decoding test fixture
# detector training runs on GitHub Actions: "Train part detector" workflow (about 4 h on CPU)
pytest -v
```

## Project structure

```
app.py, ocr_utils.py, speech_utils.py, detector_utils.py   Streamlit web app
generate_data.py, train_models.py, export_android.py      ML data, training, export to Android
detector/                                                 detector dataset prep + training
make_demo_images.py, sample_images/                       50 demo boards
android/                                                  offline Android app (Kotlin + Jetpack Compose)
  app/src/main/java/.../core/                             on-device ML engine (plain Java, unit-tested)
tests/                                                    pytest suite
.github/workflows/                                        CI: tests, APK build, emulator test, detector training
```

## Limitations

* Repair records and board histories are synthetic. With real job records, the same pipeline retrains unchanged.
* The detector struggles with the tiniest SMD parts on very dense boards (tiling the photo, a larger model or a GPU
  would help), and inductors have very few examples. It also sometimes boxes the yellow asset tag as a "connector".
* Voice notes are English only. Vosk and Whisper both support Urdu, which is a natural next step.
