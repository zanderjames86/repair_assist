package com.pcbrepair.assistant.core;

import com.google.gson.Gson;

import java.io.Reader;
import java.util.List;
import java.util.Map;

/**
 * ML model 2 on the device: a scikit-learn Random Forest (classifier or
 * regressor) exported as arrays of tree nodes. The one-hot encoding of the
 * categorical inputs matches the sklearn ColumnTransformer exactly.
 */
public final class RandomForest {

    static final class Tree {
        int[] feature;
        double[] threshold;
        int[] left;
        int[] right;
        double[] value;
    }

    static final class Encoder {
        Map<String, List<String>> categorical;
        List<String> numeric;
    }

    static final class Json {
        Encoder encoder;
        Tree[] trees;
    }

    private final Tree[] trees;
    private final List<String> boardTypes;
    private final List<String> faultCategories;
    private final int width;

    public RandomForest(Reader json) {
        Json j = new Gson().fromJson(json, Json.class);
        trees = j.trees;
        boardTypes = j.encoder.categorical.get("board_type");
        faultCategories = j.encoder.categorical.get("fault_category");
        width = boardTypes.size() + faultCategories.size() + 2;
    }

    public List<String> boardTypes() { return boardTypes; }

    /** Same column order as sklearn: one-hot(board) + one-hot(fault) + age + previous repairs. */
    float[] encode(String boardType, String faultCategory, int ageYears, int previousRepairs) {
        float[] x = new float[width];
        int b = boardTypes.indexOf(boardType);
        if (b >= 0) x[b] = 1f;  // unknown category -> all zeros (handle_unknown="ignore")
        int f = faultCategories.indexOf(faultCategory);
        if (f >= 0) x[boardTypes.size() + f] = 1f;
        x[width - 2] = ageYears;
        x[width - 1] = previousRepairs;
        return x;
    }

    /** Mean of the leaf values: P(BER) for the classifier, hours for the regressor. */
    public double predict(String boardType, String faultCategory, int ageYears, int previousRepairs) {
        float[] x = encode(boardType, faultCategory, ageYears, previousRepairs);
        double sum = 0;
        for (Tree t : trees) {
            int node = 0;
            while (t.left[node] != -1) {
                // sklearn compares the float32 input against the threshold
                node = (x[t.feature[node]] <= t.threshold[node]) ? t.left[node] : t.right[node];
            }
            sum += t.value[node];
        }
        return sum / trees.length;
    }
}
