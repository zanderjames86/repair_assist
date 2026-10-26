package com.pcbrepair.assistant.core;

import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Pulls board type, revision and serial number out of raw OCR text. */
public final class LabelParser {

    public static final class Label {
        public final String boardType;
        public final String revision;
        public final String serial;
        Label(String boardType, String revision, String serial) {
            this.boardType = boardType; this.revision = revision; this.serial = serial;
        }
    }

    private static final Pattern SERIAL =
            Pattern.compile("S\\s*[/I1L|]?\\s*N\\s*[:.\\-]?\\s*([0-9OQ]{4,6})");
    private static final Pattern REV = Pattern.compile("REV\\.?\\s*([A-Z])\\b");
    private static final Pattern BOARD = Pattern.compile("[A-Z0-9]{2,}\\s*[-_]\\s*[A-Z0-9]{2,}");

    private LabelParser() {}

    public static Label parse(String text, List<String> knownBoards) {
        String up = text == null ? "" : text.toUpperCase(Locale.ROOT);
        String serial = null;
        Matcher m = SERIAL.matcher(up);
        if (m.find()) serial = m.group(1).replace('O', '0').replace('Q', '0');
        String rev = null;
        m = REV.matcher(up);
        if (m.find()) rev = m.group(1);
        String board = null;
        m = BOARD.matcher(up);
        while (board == null && m.find()) {
            String tok = m.group().replaceAll("\\s*[-_]\\s*", "-");
            board = closest(tok, knownBoards, 0.6);
            if (board == null) {   // OCR often swaps letters for look-alike digits
                String fixed = tok.replace('0', 'O').replace('1', 'I').replace('5', 'S').replace('8', 'B');
                board = closest(fixed, knownBoards, 0.6);
            }
        }
        return new Label(board, rev, serial);
    }

    /** Best match with normalised similarity >= cutoff, or null. */
    static String closest(String word, List<String> candidates, double cutoff) {
        String best = null;
        double bestScore = cutoff;
        for (String c : candidates) {
            double s = similarity(word, c);
            if (s >= bestScore) { bestScore = s; best = c; }
        }
        return best;
    }

    static double similarity(String a, String b) {
        if (a.isEmpty() && b.isEmpty()) return 1.0;
        int[] prev = new int[b.length() + 1];
        int[] cur = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) prev[j] = j;
        for (int i = 1; i <= a.length(); i++) {
            cur[0] = i;
            for (int j = 1; j <= b.length(); j++) {
                int cost = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                cur[j] = Math.min(Math.min(cur[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
            }
            int[] t = prev; prev = cur; cur = t;
        }
        return 1.0 - (double) prev[b.length()] / Math.max(a.length(), b.length());
    }
}
