package com.pcbrepair.assistant.core;

import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;

import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * ML model 1 on the device: TF-IDF + multinomial Logistic Regression.
 * Re-implements scikit-learn's TfidfVectorizer (lowercase, token pattern
 * (?u)\b\w\w+\b, stop words, 1-2 grams, sublinear tf, l2 norm) and
 * LogisticRegression.predict_proba (softmax) using weights exported to JSON.
 */
public final class FaultClassifier {

    private static final Pattern TOKEN =
            Pattern.compile("\\b\\w\\w+\\b", Pattern.UNICODE_CHARACTER_CLASS);

    static final class Json {
        String[] classes;
        @SerializedName("stop_words") String[] stopWords;
        @SerializedName("ngram_max") int ngramMax;
        Map<String, Integer> vocabulary;
        double[] idf;
        double[][] coef;
        double[] intercept;
    }

    public static final class Contribution {
        public final String term;
        public final double weight;
        Contribution(String term, double weight) { this.term = term; this.weight = weight; }
    }

    private final String[] classes;
    private final Set<String> stopWords;
    private final Map<String, Integer> vocab;
    private final double[] idf;
    private final double[][] coef;
    private final double[] intercept;
    private final String[] termByIndex;

    public FaultClassifier(Reader json) {
        Json j = new Gson().fromJson(json, Json.class);
        if (j.ngramMax != 2) throw new IllegalArgumentException("only 1-2 grams supported");
        classes = j.classes;
        stopWords = new HashSet<>(List.of(j.stopWords));
        vocab = j.vocabulary;
        idf = j.idf;
        coef = j.coef;
        intercept = j.intercept;
        termByIndex = new String[idf.length];
        for (Map.Entry<String, Integer> e : vocab.entrySet()) termByIndex[e.getValue()] = e.getKey();
    }

    public String[] classes() { return classes.clone(); }

    /** Tokens after lowercasing and stop-word removal (sklearn's analyzer). */
    static List<String> terms(String note, Set<String> stopWords) {
        List<String> tokens = new ArrayList<>();
        Matcher m = TOKEN.matcher(note.toLowerCase(Locale.ROOT));
        while (m.find()) {
            String t = m.group();
            if (!stopWords.contains(t)) tokens.add(t);
        }
        List<String> out = new ArrayList<>(tokens);
        for (int i = 0; i + 1 < tokens.size(); i++) out.add(tokens.get(i) + " " + tokens.get(i + 1));
        return out;
    }

    /** Sparse TF-IDF vector: feature index -> l2-normalised weight. */
    Map<Integer, Double> vectorize(String note) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (String t : terms(note == null ? "" : note, stopWords)) {
            Integer idx = vocab.get(t);
            if (idx != null) counts.merge(idx, 1, Integer::sum);
        }
        Map<Integer, Double> x = new HashMap<>();
        double norm = 0;
        for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
            double v = (1.0 + Math.log(e.getValue())) * idf[e.getKey()];
            x.put(e.getKey(), v);
            norm += v * v;
        }
        if (norm > 0) {
            double n = Math.sqrt(norm);
            x.replaceAll((k, v) -> v / n);
        }
        return x;
    }

    public double[] predictProba(String note) {
        Map<Integer, Double> x = vectorize(note);
        double[] z = new double[classes.length];
        double max = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < classes.length; c++) {
            double s = intercept[c];
            for (Map.Entry<Integer, Double> e : x.entrySet()) s += coef[c][e.getKey()] * e.getValue();
            z[c] = s;
            max = Math.max(max, s);
        }
        double sum = 0;
        for (int c = 0; c < z.length; c++) { z[c] = Math.exp(z[c] - max); sum += z[c]; }
        for (int c = 0; c < z.length; c++) z[c] /= sum;
        return z;
    }

    public static int argmax(double[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
        return best;
    }

    /** Terms in the note that pushed the model towards class {@code classIndex}. */
    public List<Contribution> explain(String note, int classIndex, int topK) {
        List<Contribution> out = new ArrayList<>();
        for (Map.Entry<Integer, Double> e : vectorize(note).entrySet()) {
            double w = e.getValue() * coef[classIndex][e.getKey()];
            if (w > 0) out.add(new Contribution(termByIndex[e.getKey()], w));
        }
        out.sort((a, b) -> Double.compare(b.weight, a.weight));
        return out.size() > topK ? new ArrayList<>(out.subList(0, topK)) : out;
    }
}
