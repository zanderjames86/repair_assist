package com.pcbrepair.assistant.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

public class LabelAndYoloTest {

    static final List<String> BOARDS = Arrays.asList(
            "PSU-CTRL", "MCU-MAIN", "SENSOR-IF", "RF-COMM", "MOTOR-DRV", "DISPLAY-DRV");

    // ---------------- OCR label parsing
    @Test public void parsesCleanLabel() {
        LabelParser.Label l = LabelParser.parse("PSU-CTRL\nREV A\nS/N 2291", BOARDS);
        assertEquals("PSU-CTRL", l.boardType);
        assertEquals("A", l.revision);
        assertEquals("2291", l.serial);
    }

    @Test public void toleratesTypicalOcrErrors() {
        LabelParser.Label l = LabelParser.parse("_®\n1 RF - C0MM\nREVC\nS|N: 5O18", BOARDS);
        assertEquals("5018", l.serial);
        assertEquals("C", l.revision);
        assertEquals("RF-COMM", l.boardType);
    }

    @Test public void fuzzyBoardMatch() {
        assertEquals("SENSOR-IF", LabelParser.parse("SENS0R-IF".replace('0', 'O'), BOARDS).boardType);
        assertEquals("DISPLAY-DRV", LabelParser.parse("DISPLAV-DRV s/n 9043", BOARDS).boardType);
    }

    @Test public void lowerCaseAndSeparators() {
        LabelParser.Label l = LabelParser.parse("mcu_main rev. b sn-3107", BOARDS);
        assertEquals("MCU-MAIN", l.boardType);
        assertEquals("B", l.revision);
        assertEquals("3107", l.serial);
    }

    @Test public void nothingFound() {
        LabelParser.Label l = LabelParser.parse("hello world", BOARDS);
        assertNull(l.boardType); assertNull(l.revision); assertNull(l.serial);
        l = LabelParser.parse(null, BOARDS);
        assertNull(l.serial);
    }

    @Test public void similarityBounds() {
        assertEquals(1.0, LabelParser.similarity("ABC", "ABC"), 0);
        assertEquals(0.0, LabelParser.similarity("ABC", "XYZ"), 0);
    }

    // ---------------- YOLO pre/post-processing
    @Test public void letterboxLandscape() {
        YoloDecoder.Letterbox lb = new YoloDecoder.Letterbox(1280, 640, 640);
        assertEquals(0.5f, lb.scale, 1e-6);
        assertEquals(640, lb.newW); assertEquals(320, lb.newH);
        assertEquals(0, lb.padX); assertEquals(160, lb.padY);
    }

    @Test public void tensorLayoutAndPadding() {
        YoloDecoder.Letterbox lb = new YoloDecoder.Letterbox(2, 1, 4);   // -> 4x2 image, 1 row pad top
        int[] px = new int[lb.newW * lb.newH];
        Arrays.fill(px, 0xFFFF0000);                                    // pure red
        float[] t = lb.toTensor(px);
        int plane = 16;
        assertEquals(114f / 255f, t[0], 1e-6);                          // padded corner is grey
        int i = lb.padY * 4;                                            // first image pixel
        assertEquals(1f, t[i], 1e-6);                                   // R
        assertEquals(0f, t[plane + i], 1e-6);                           // G
        assertEquals(0f, t[2 * plane + i], 1e-6);                       // B
    }

    static float[] output(int nc, float[][] anchors) {
        int n = anchors.length;
        float[] out = new float[(4 + nc) * n];
        for (int a = 0; a < n; a++) for (int k = 0; k < 4 + nc; k++) out[k * n + a] = anchors[a][k];
        return out;
    }

    @Test public void decodeMapsBackToOriginalPixels() {
        YoloDecoder.Letterbox lb = new YoloDecoder.Letterbox(1280, 640, 640);  // scale .5, padY 160
        float[] out = output(2, new float[][]{{320, 320, 100, 50, 0.9f, 0.1f}});
        List<YoloDecoder.Detection> d = YoloDecoder.decode(out, 2, 1, lb, 1280, 640, 0.25f, 0.45f);
        assertEquals(1, d.size());
        YoloDecoder.Detection x = d.get(0);
        assertEquals(0, x.classId);
        assertEquals(540f, x.left, 1e-3);
        assertEquals(270f, x.top, 1e-3);
        assertEquals(740f, x.right, 1e-3);
        assertEquals(370f, x.bottom, 1e-3);
    }

    @Test public void scoreThresholdAndNms() {
        YoloDecoder.Letterbox lb = new YoloDecoder.Letterbox(640, 640, 640);
        float[] out = output(2, new float[][]{
                {100, 100, 40, 40, 0.90f, 0.0f},   // keep
                {102, 101, 40, 40, 0.80f, 0.0f},   // overlaps same class -> suppressed
                {102, 101, 40, 40, 0.00f, 0.70f},  // overlaps but other class -> keep
                {400, 400, 40, 40, 0.20f, 0.10f},  // below threshold
                {500, 500, 30, 30, 0.60f, 0.0f},   // separate -> keep
        });
        List<YoloDecoder.Detection> d = YoloDecoder.decode(out, 2, 5, lb, 640, 640, 0.25f, 0.45f);
        assertEquals(3, d.size());
        assertEquals(0.90f, d.get(0).score, 1e-6);
        assertTrue(d.stream().anyMatch(x -> x.classId == 1));
    }

    @Test public void nmsIsStableOnEmptyInput() {
        assertTrue(YoloDecoder.nms(new ArrayList<>(), 0.5f).isEmpty());
    }

    @Test(expected = IllegalArgumentException.class)
    public void decodeRejectsWrongShape() {
        YoloDecoder.decode(new float[7], 2, 2, new YoloDecoder.Letterbox(10, 10, 10), 10, 10, .25f, .45f);
    }
}
