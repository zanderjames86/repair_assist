# Class demo script (about 6 minutes)

## Before class
- Install the APK on the tablet (see README). Open the app once. It works without Wi-Fi, so turn on
  **airplane mode** and say so: it proves the air-gapped design.
- Mirror the tablet to the projector (USB-C to HDMI, or the classroom's casting option before switching on airplane mode).
- Backup: the Streamlit web app link, open on the laptop.
- Optional: print the `board_1686.jpg` photo (Zybo board with its yellow tag) and point the tablet camera at it.

## 1. The problem (30 s)
"Repair technicians have their hands busy, so typing job cards is slow. Deciding whether a board is worth
repairing depends on experience. This app does both, fully offline, on a tablet at the bench."

Open the **Models** tab: "Five techniques. Three AI models: OCR, speech-to-text, and a YOLO part detector
I fine-tuned. Two classical ML models I trained: a fault classifier and a repair-outcome Random Forest.
Everything runs on the device, and the app has no internet permission."

## 2. Live run on the Bench tab (3 min)
1. **Demo boards → `board_1686`** (a Zybo FPGA board).
   - **OCR (AI):** the yellow asset tag is read: RF-COMM, Rev A, S/N 1686. The board's history fills in.
   - **Part detector (AI):** boxes appear on ICs, connectors, capacitors and switches, with counts per type.
     "This model is YOLOv8n, fine-tuned on 137 real board photos. This board was never in its training set."
2. **Voice note (AI):** tap the mic and say:
   > "No output on the five volt rail, regulator U3 is hot."

   Vosk transcribes it offline. If the mic misbehaves, tap the first example note.
3. **Fault diagnosis (ML 1):** Power supply, with the probability bars and the words that drove the decision.
4. **Recommendation (ML 2):** ✅ Repairable, about 35% BER risk.
5. **Contrast:** replace the note with the corrosion example
   (*"Green deposits on the pins, salt residue across the lower edge, moisture ingress."*).
   The fault becomes **Corrosion** and the recommendation flips to **⛔ BER** (about 71% risk, about 7.6 h).
   "Salt-water corrosion on a complex RF board is exactly when repair stops being worth it."
6. **Save**, then open the **Log** tab and show the CSV export (saves to tablet storage or USB).

## 3. Models tab (1.5 min)
- **Fault classifier:** "I compared Naive Bayes, Logistic Regression and Linear SVM with 5-fold
  cross-validation, about 95% accuracy. I chose Logistic Regression for its probabilities and
  explanations." Show the confusion matrix.
- **Random Forest:** ROC AUC 0.82. Repair-hours error 0.43 h, against 1.24 h for guessing the average.
  "Board age and fault type drive scrap decisions, which matches how repair engineers think."
- **Part detector:** mAP@50 0.35 overall on 22 unseen test boards. ICs 0.63, switches 0.59 and connectors 0.54
  are strong; tiny SMD resistors and capacitors are hard for a nano model on a CPU budget. "Fine-tuned from COCO
  weights for 4.8 hours on GitHub's free CPU runners, no GPU."
- "On the phone, the classical models run from exported weights. Automated tests check that the phone's
  answers match Python to six decimal places on 600 cases."

## 4. Engineering and honesty (1 min)
- Testing: 41 Python tests, 19 JVM unit tests and 7 end-to-end tests on an Android emulator run on every
  push in GitHub Actions. On the emulator, ML Kit read 94% of the tags, and offline speech recognised
  21 of 23 key words and diagnosed 4 of 4 spoken notes correctly.
- OCR: 49/50 tags read correctly once the tag is located and cleaned, against 15/50 for raw OCR.
  The miss is a yellow tag on a yellow motherboard.
- Limits: the repair records are synthetic (no sensitive data), the tiniest SMD parts are hard to detect,
  and voice is English only. Next steps: real job data, image tiling, Urdu voice notes.

## Likely questions
- **Why classical ML for the fault classifier?** Short notes and 600 records. A linear model on TF-IDF is
  accurate, explainable, tiny (120 KB) and runs anywhere. Deep learning is used where it's needed:
  vision and speech.
- **How do the Python models run on Android?** Their learned parameters (vocabulary, IDF weights,
  coefficients, tree nodes) are exported to JSON, and a small Java engine replays the same maths.
  The parity tests prove it.
- **Why is BER accuracy only 75%?** Scrap decisions have real randomness. ROC AUC 0.82 shows the model
  ranks risk well, so the probability is more useful than the yes/no.
- **Why Vosk instead of Whisper on the phone?** Vosk runs in real time on a tablet CPU with a 40 MB model,
  fully offline. The web app uses Whisper.
