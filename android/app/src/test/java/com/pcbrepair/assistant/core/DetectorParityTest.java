package com.pcbrepair.assistant.core;

import static org.junit.Assert.assertEquals;

import com.google.gson.Gson;

import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.Test;

/** The phone decodes YOLO output exactly like the Python web app (detector_utils.decode). */
public class DetectorParityTest {

    static final class Det { int c; float s; float[] b; }
    static final class Case { int nc, n, size, w, h; float[] out; Det[] dets; }

    @Test
    public void decodeMatchesPython() {
        Case[] cases = new Gson().fromJson(new InputStreamReader(
                getClass().getResourceAsStream("/detector_parity.json"), StandardCharsets.UTF_8), Case[].class);
        assertEquals(3, cases.length);
        for (Case c : cases) {
            YoloDecoder.Letterbox lb = new YoloDecoder.Letterbox(c.w, c.h, c.size);
            List<YoloDecoder.Detection> d = YoloDecoder.decode(c.out, c.nc, c.n, lb, c.w, c.h, 0.30f, 0.45f);
            assertEquals("detections", c.dets.length, d.size());
            for (int i = 0; i < d.size(); i++) {
                YoloDecoder.Detection x = d.get(i);
                Det e = c.dets[i];
                assertEquals("class #" + i, e.c, x.classId);
                assertEquals(e.s, x.score, 1e-6);
                assertEquals(e.b[0], x.left, 0.05);
                assertEquals(e.b[1], x.top, 0.05);
                assertEquals(e.b[2], x.right, 0.05);
                assertEquals(e.b[3], x.bottom, 0.05);
            }
        }
    }
}
