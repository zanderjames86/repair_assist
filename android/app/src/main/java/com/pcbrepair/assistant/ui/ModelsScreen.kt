package com.pcbrepair.assistant.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.gson.JsonObject
import com.pcbrepair.assistant.BenchViewModel
import com.pcbrepair.assistant.Engine
import com.pcbrepair.assistant.faultName

private val FEATURE_NAMES = mapOf(
    "age_years" to "Board age", "fault_category" to "Fault category",
    "previous_repairs" to "Previous repairs", "board_type" to "Board type",
)

@Composable
fun ModelsScreen(vm: BenchViewModel, engine: Engine) {
    val m = engine.metrics
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp).testTag("models_screen"),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        StepCard(null, "What runs on this device") {
            Text("Everything runs offline. The app has no internet permission.", fontWeight = FontWeight.SemiBold)
            KindBadge("AI", "OCR: ML Kit text recognition (bundled model)")
            KindBadge("AI", "Speech-to-text: Vosk small English model")
            KindBadge("AI", "Part detector: YOLOv8n, fine-tuned by us on open PCB photos")
            KindBadge("ML", "Fault classifier: TF-IDF + Logistic Regression")
            KindBadge("ML", "Repair predictor: Random Forest classifier + regressor")
            Caption("AI = deep-learning models; ML = classical models we trained on ${m.get("n_records").asInt} synthetic repair records. The phone runs the exported model weights and is unit-tested to match the Python originals to 6 decimal places.")
        }
        FaultCard(m)
        OutcomeCard(m)
        PartsCard(engine.partMetrics)
    }
}

@Composable
private fun FaultCard(m: JsonObject) {
    val fc = m.getAsJsonObject("fault_classifier")
    StepCard(null, "Fault classifier") {
        KindBadge("ML", "Text → fault category, 5-fold cross-validation")
        Row(Modifier.fillMaxWidth()) {
            listOf("Model" to 1.6f, "CV F1" to 0.8f, "Test acc." to 0.9f).forEach { (h, w) ->
                Text(h, Modifier.weight(w), fontSize = 12.sp, color = Ink2)
            }
        }
        fc.getAsJsonArray("comparison").forEach { e ->
            val o = e.asJsonObject
            val chosen = o.get("model").asString == fc.get("chosen").asString
            Row(Modifier.fillMaxWidth()) {
                Text(o.get("model").asString + if (chosen) "  ✓" else "", Modifier.weight(1.6f),
                    fontWeight = if (chosen) FontWeight.SemiBold else FontWeight.Normal)
                Text("%.3f".format(o.get("cv_f1_mean").asDouble), Modifier.weight(0.8f))
                Text("%.3f".format(o.get("test_accuracy").asDouble), Modifier.weight(0.9f))
            }
        }
        Text("Confusion matrix (test set)", fontWeight = FontWeight.SemiBold)
        val labels = fc.getAsJsonArray("labels").map { faultName(it.asString) }
        val cm = fc.getAsJsonArray("confusion_matrix").map { r -> r.asJsonArray.map { it.asInt } }
        ConfusionMatrix(labels, cm)
    }
}

@Composable
private fun ConfusionMatrix(labels: List<String>, cm: List<List<Int>>) {
    val max = cm.flatten().maxOrNull()?.coerceAtLeast(1) ?: 1
    Column(Modifier.fillMaxWidth()) {
        cm.forEachIndexed { i, row ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(labels[i], Modifier.width(118.dp), fontSize = 11.sp, color = Ink2)
                row.forEach { v ->
                    val t = v.toFloat() / max
                    Box(
                        Modifier.weight(1f).aspectRatio(1f).padding(1.dp)
                            .background(lerp(Color(0xFFEEF4FD), Color(0xFF0D366B), t)),
                        contentAlignment = Alignment.Center,
                    ) { Text("$v", fontSize = 11.sp, color = if (t > 0.5f) Color.White else Ink) }
                }
            }
        }
        Row {
            Box(Modifier.width(118.dp))
            labels.forEach { Text(it.take(5), Modifier.weight(1f), fontSize = 9.sp, color = Ink2, textAlign = TextAlign.Center) }
        }
        Caption("Rows = actual, columns = predicted.")
    }
}

@Composable
private fun OutcomeCard(m: JsonObject) {
    val oc = m.getAsJsonObject("outcome")
    val hr = m.getAsJsonObject("hours")
    StepCard(null, "Repair outcome predictor") {
        KindBadge("ML", "Random Forest on board type, age, previous repairs, fault")
        Row(Modifier.fillMaxWidth()) {
            Metric("BER ROC AUC", "%.2f".format(oc.get("test_roc_auc").asDouble), Modifier.weight(1f))
            Metric("BER accuracy", "%.0f%%".format(oc.get("test_accuracy").asDouble * 100), Modifier.weight(1f))
        }
        Caption("Baseline (always 'Repairable'): %.0f%%".format(oc.get("baseline_accuracy").asDouble * 100))
        Row(Modifier.fillMaxWidth()) {
            Metric("Hours MAE", "%.2f h".format(hr.get("test_mae").asDouble), Modifier.weight(1f))
            Metric("Hours R²", "%.2f".format(hr.get("test_r2").asDouble), Modifier.weight(1f))
        }
        Caption("Baseline (average guess): %.2f h".format(hr.get("baseline_mae").asDouble))
        HorizontalDivider()
        Text("What drives BER decisions", fontWeight = FontWeight.SemiBold)
        importance(oc.getAsJsonObject("feature_importance"))
        Text("What drives repair time", fontWeight = FontWeight.SemiBold)
        importance(hr.getAsJsonObject("feature_importance"))
    }
}

@Composable
private fun importance(o: JsonObject) {
    val entries = o.entrySet().map { it.key to it.value.asDouble }.sortedByDescending { it.second }
    val max = entries.maxOfOrNull { it.second } ?: 1.0
    entries.forEach { (k, v) -> BarRow(FEATURE_NAMES[k] ?: k, (v / max).toFloat(), "%.0f%%".format(v * 100)) }
}

@Composable
private fun PartsCard(p: JsonObject?) {
    StepCard(null, "Part detector") {
        KindBadge("AI", "YOLOv8n object detection, fine-tuned on open PCB datasets")
        if (p == null) { Caption("Detector not bundled in this build."); return@StepCard }
        val met = p.getAsJsonObject("metrics")
        if (met != null) Row(Modifier.fillMaxWidth()) {
            Metric("mAP@50", "%.2f".format(met.get("map50").asDouble), Modifier.weight(1f))
            Metric("Precision", "%.2f".format(met.get("precision").asDouble), Modifier.weight(1f))
            Metric("Recall", "%.2f".format(met.get("recall").asDouble), Modifier.weight(1f))
        }
        val per = p.getAsJsonObject("per_class_ap50")
        if (per != null) {
            Text("mAP@50 per part type", fontWeight = FontWeight.SemiBold)
            per.entrySet().sortedByDescending { it.value.asDouble }.forEach {
                BarRow(it.key, it.value.asDouble.toFloat(), "%.2f".format(it.value.asDouble))
            }
        }
        p.get("dataset")?.let { Caption(it.asString) }
    }
}
