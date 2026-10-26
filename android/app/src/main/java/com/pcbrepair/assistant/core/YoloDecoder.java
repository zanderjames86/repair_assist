package com.pcbrepair.assistant.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Pre- and post-processing for the YOLOv8 part detector (ONNX).
 * Input: letterboxed RGB image, CHW float in [0,1], size x size.
 * Output: [1, 4 + numClasses, N] with (cx, cy, w, h) in input pixels,
 * followed by per-class scores. Applies a score threshold and per-class NMS.
 */
public final class YoloDecoder {

    public static final class Detection {
        public final int classId;
        public final float score;
        public final float left, top, right, bottom;   // original image pixels
        public Detection(int classId, float score, float left, float top, float right, float bottom) {
            this.classId = classId; this.score = score;
            this.left = left; this.top = top; this.right = right; this.bottom = bottom;
        }
        float area() { return Math.max(0, right - left) * Math.max(0, bottom - top); }
    }

    /** Where the original image sits inside the square model input. */
    public static final class Letterbox {
        public final float scale;
        public final int newW, newH, padX, padY, size;
        public Letterbox(int origW, int origH, int size) {
            this.size = size;
            scale = Math.min((float) size / origW, (float) size / origH);
            newW = Math.round(origW * scale);
            newH = Math.round(origH * scale);
            padX = (size - newW) / 2;
            padY = (size - newH) / 2;
        }

        /**
         * Builds the CHW float tensor from the already-resized image pixels
         * (ARGB ints, newW x newH). Padding is grey 114, as in training.
         */
        public float[] toTensor(int[] resizedArgb) {
            if (resizedArgb.length != newW * newH) throw new IllegalArgumentException("pixel count");
            int plane = size * size;
            float[] t = new float[3 * plane];
            float grey = 114f / 255f;
            java.util.Arrays.fill(t, grey);
            for (int y = 0; y < newH; y++) {
                for (int x = 0; x < newW; x++) {
                    int p = resizedArgb[y * newW + x];
                    int i = (y + padY) * size + (x + padX);
                    t[i] = ((p >> 16) & 0xFF) / 255f;
                    t[plane + i] = ((p >> 8) & 0xFF) / 255f;
                    t[2 * plane + i] = (p & 0xFF) / 255f;
                }
            }
            return t;
        }
    }

    private YoloDecoder() {}

    public static List<Detection> decode(float[] out, int numClasses, int numAnchors, Letterbox lb,
                                         int origW, int origH, float scoreThreshold, float iouThreshold) {
        if (out.length != (4 + numClasses) * numAnchors) throw new IllegalArgumentException("output size");
        List<Detection> candidates = new ArrayList<>();
        for (int a = 0; a < numAnchors; a++) {
            int best = -1;
            float bestScore = scoreThreshold;
            for (int c = 0; c < numClasses; c++) {
                float s = out[(4 + c) * numAnchors + a];
                if (s > bestScore) { bestScore = s; best = c; }
            }
            if (best < 0) continue;
            float cx = out[a], cy = out[numAnchors + a];
            float w = out[2 * numAnchors + a], h = out[3 * numAnchors + a];
            float l = clamp((cx - w / 2 - lb.padX) / lb.scale, origW);
            float t = clamp((cy - h / 2 - lb.padY) / lb.scale, origH);
            float r = clamp((cx + w / 2 - lb.padX) / lb.scale, origW);
            float b = clamp((cy + h / 2 - lb.padY) / lb.scale, origH);
            if (r - l < 1 || b - t < 1) continue;
            candidates.add(new Detection(best, bestScore, l, t, r, b));
        }
        return nms(candidates, iouThreshold);
    }

    private static float clamp(float v, int max) { return Math.max(0, Math.min(v, max)); }

    static float iou(Detection a, Detection b) {
        float l = Math.max(a.left, b.left), t = Math.max(a.top, b.top);
        float r = Math.min(a.right, b.right), btm = Math.min(a.bottom, b.bottom);
        float inter = Math.max(0, r - l) * Math.max(0, btm - t);
        float union = a.area() + b.area() - inter;
        return union <= 0 ? 0 : inter / union;
    }

    /** Greedy per-class non-maximum suppression, highest score first. */
    public static List<Detection> nms(List<Detection> dets, float iouThreshold) {
        List<Detection> sorted = new ArrayList<>(dets);
        sorted.sort((x, y) -> Float.compare(y.score, x.score));
        List<Detection> keep = new ArrayList<>();
        boolean[] removed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (removed[i]) continue;
            Detection d = sorted.get(i);
            keep.add(d);
            for (int j = i + 1; j < sorted.size(); j++) {
                Detection e = sorted.get(j);
                if (!removed[j] && e.classId == d.classId && iou(d, e) > iouThreshold) removed[j] = true;
            }
        }
        return keep;
    }
}
