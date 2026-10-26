package com.pcbrepair.assistant.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.google.gson.Gson;

import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.junit.BeforeClass;
import org.junit.Test;

/**
 * The phone must give the same answers as the trained Python models.
 * parity.json holds hundreds of inputs with scikit-learn's exact outputs.
 */
public class ParityTest {

    static final double TOL = 1e-6;
    static FaultClassifier fault;
    static RandomForest outcome, hours;
    static Parity parity;

    static final class NoteCase { String note; double[] proba; }
    static final class TabCase {
        String board_type, fault_category; int age_years, previous_repairs; double p_ber, hours;
    }
    static final class Parity { NoteCase[] notes; TabCase[] tabular; }

    static Path assets() {
        for (String p : new String[]{"src/main/assets/models", "app/src/main/assets/models",
                "android/app/src/main/assets/models"}) {
            if (Files.isDirectory(Paths.get(p))) return Paths.get(p);
        }
        throw new IllegalStateException("assets/models not found from " + Paths.get("").toAbsolutePath());
    }

    static Reader open(String name) throws Exception {
        return Files.newBufferedReader(assets().resolve(name), StandardCharsets.UTF_8);
    }

    @BeforeClass
    public static void load() throws Exception {
        fault = new FaultClassifier(open("fault_lr.json"));
        outcome = new RandomForest(open("outcome_rf.json"));
        hours = new RandomForest(open("hours_rf.json"));
        parity = new Gson().fromJson(new InputStreamReader(
                ParityTest.class.getResourceAsStream("/parity.json"), StandardCharsets.UTF_8), Parity.class);
    }

    @Test
    public void faultClassifierMatchesSklearn() {
        assertTrue(parity.notes.length >= 300);
        for (NoteCase c : parity.notes) {
            double[] p = fault.predictProba(c.note);
            for (int k = 0; k < p.length; k++) {
                assertEquals("note: " + c.note, c.proba[k], p[k], TOL);
            }
        }
    }

    @Test
    public void probabilitiesSumToOne() {
        for (NoteCase c : parity.notes) {
            double s = 0;
            for (double v : fault.predictProba(c.note)) s += v;
            assertEquals(1.0, s, 1e-9);
        }
    }

    @Test
    public void outcomeForestMatchesSklearn() {
        assertTrue(parity.tabular.length >= 300);
        for (TabCase c : parity.tabular) {
            assertEquals(c.board_type, c.p_ber,
                    outcome.predict(c.board_type, c.fault_category, c.age_years, c.previous_repairs), TOL);
        }
    }

    @Test
    public void hoursForestMatchesSklearn() {
        for (TabCase c : parity.tabular) {
            assertEquals(c.board_type, c.hours,
                    hours.predict(c.board_type, c.fault_category, c.age_years, c.previous_repairs), TOL);
        }
    }

    @Test
    public void explanationPointsAtCorrosionWords() {
        String note = "Green deposits on the pins, salt residue across the lower edge, moisture ingress.";
        double[] p = fault.predictProba(note);
        int k = FaultClassifier.argmax(p);
        assertEquals("corrosion", fault.classes()[k]);
        List<FaultClassifier.Contribution> why = fault.explain(note, k, 5);
        assertTrue(why.size() >= 3);
        for (int i = 1; i < why.size(); i++) assertTrue(why.get(i - 1).weight >= why.get(i).weight);
    }

    @Test
    public void emptyNoteGivesValidDistribution() {
        double[] p = fault.predictProba("");
        assertEquals(6, p.length);
        for (double v : p) assertTrue(v > 0 && v < 1);
        double[] q = fault.predictProba(null);
        assertEquals(p[0], q[0], 0);
    }
}
