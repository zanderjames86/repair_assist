package com.pcbrepair.assistant.ui

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.pcbrepair.assistant.BenchViewModel
import com.pcbrepair.assistant.Engine
import com.pcbrepair.assistant.Images
import com.pcbrepair.assistant.faultName
import java.io.File

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun BenchScreen(vm: BenchViewModel, engine: Engine) {
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        PhotoStep(vm, engine)
        BoardStep(vm, engine)
        PartsStep(vm)
        VoiceStep(vm, engine)
        val d = vm.diagnose()
        if (d != null && vm.board() != null) {
            DiagnosisStep(vm, d)
            RecommendationStep(vm, d)
            Button(
                onClick = { vm.save() }, modifier = Modifier.fillMaxWidth().testTag("save"),
            ) { Icon(Icons.Filled.Save, null); Text("  Save job card") }
        } else {
            Caption(if (vm.board() == null) "Scan or select a board to continue." else "Add a voice note (or type one) to get the diagnosis.")
        }
    }
}

// ---------------------------------------------------------------- 1. photo + OCR
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PhotoStep(vm: BenchViewModel, engine: Engine) {
    val context = LocalContext.current
    var showDemo by remember { mutableStateOf(false) }
    var pendingUri by remember { mutableStateOf<Uri?>(null) }

    val camera = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val uri = pendingUri
        if (ok && uri != null) vm.loadUri(uri, "camera photo")
    }
    val gallery = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) vm.loadUri(uri, "gallery photo")
    }

    StepCard(1, "Photograph the board") {
        KindBadge("AI", "OCR reads the board label (ML Kit, on-device)")
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { showDemo = true }, modifier = Modifier.testTag("demo_button")) {
                Icon(Icons.Filled.Memory, null); Text("  Demo boards")
            }
            OutlinedButton(onClick = {
                val dir = File(context.cacheDir, "photos").apply { mkdirs() }
                val file = File(dir, "board_${System.currentTimeMillis()}.jpg")
                val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
                pendingUri = uri
                runCatching { camera.launch(uri) }.onFailure { vm.message = "No camera app available" }
            }) { Icon(Icons.Filled.CameraAlt, null); Text("  Camera") }
            OutlinedButton(onClick = { gallery.launch("image/*") }) {
                Icon(Icons.Filled.Image, null); Text("  Gallery")
            }
        }

        val photo = vm.photo
        if (photo != null) {
            Box(Modifier.fillMaxWidth().aspectRatio(photo.width.toFloat() / photo.height)) {
                Image(photo.asImageBitmap(), "Board photo", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
                PartOverlay(vm, photo.width, photo.height)
            }
            Caption(vm.photoName)
        }
        if (vm.ocrBusy) Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(Modifier.size(20.dp)); Text("  Reading label...")
        }
        val label = vm.label
        if (label != null && !vm.ocrBusy) {
            Caption("Raw OCR text")
            SelectionContainer {
                Text(vm.ocrText.ifBlank { "(nothing read)" }, fontFamily = FontFamily.Monospace, fontSize = 12.sp,
                    modifier = Modifier.heightIn(max = 120.dp).testTag("ocr_text"))
            }
            val ok = label.serial != null && engine.board(label.serial) != null
            Text(
                (if (ok) "✅ " else "⚠️ ") + "Board: ${label.boardType ?: "?"}   Rev: ${label.revision ?: "?"}   S/N: ${label.serial ?: "not found"}",
                fontWeight = FontWeight.SemiBold, modifier = Modifier.testTag("ocr_result"),
            )
        }
    }

    if (showDemo) {
        DemoPicker(engine, onPick = { vm.loadDemo(it); showDemo = false }, onDismiss = { showDemo = false })
    }
}

@Composable
private fun DemoPicker(engine: Engine, onPick: (String) -> Unit, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val names = remember { engine.demoImages() }
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = { TextButton(onClick = onDismiss) { Text("Close") } },
        title = { Text("Demo boards (${names.size})") },
        text = {
            LazyVerticalGrid(GridCells.Adaptive(110.dp), Modifier.height(420.dp).testTag("demo_grid")) {
                items(names) { name ->
                    val thumb = remember(name) {
                        runCatching { Images.fromAsset(context, "demo/$name") }.getOrNull()
                    }
                    Column(Modifier.padding(4.dp).clickable { onPick(name) }.testTag("demo_$name")) {
                        if (thumb != null) Image(thumb.asImageBitmap(), name,
                            Modifier.fillMaxWidth().aspectRatio(1.33f), contentScale = ContentScale.Crop)
                        Text(name.removeSuffix(".jpg"), fontSize = 11.sp)
                    }
                }
            }
        },
    )
}

@Composable
private fun PartOverlay(vm: BenchViewModel, w: Int, h: Int) {
    val parts = vm.parts
    if (parts.isEmpty()) return
    val names = vm.parts.map { it.name }.distinct().sorted()
    Canvas(Modifier.fillMaxSize()) {
        val s = size.width / w
        for (p in parts) {
            val color = PartColors[names.indexOf(p.name) % PartColors.size]
            drawRect(
                color = color,
                topLeft = Offset(p.box.left * s, p.box.top * s),
                size = Size((p.box.right - p.box.left) * s, (p.box.bottom - p.box.top) * s),
                style = Stroke(width = 2.dp.toPx()),
            )
        }
    }
}

// ---------------------------------------------------------------- 2. board history
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun BoardStep(vm: BenchViewModel, engine: Engine) {
    StepCard(2, "Board history") {
        var expanded by remember { mutableStateOf(false) }
        ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
            OutlinedTextField(
                value = vm.serial ?: "", onValueChange = {}, readOnly = true,
                label = { Text("Serial number (auto-filled by OCR)") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
                modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable).fillMaxWidth().testTag("serial_field"),
            )
            ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                engine.registry.forEach { b ->
                    DropdownMenuItem(text = { Text("${b.serial}  ·  ${b.boardType}") },
                        onClick = { vm.serial = b.serial; expanded = false })
                }
            }
        }
        val b = vm.board()
        if (b != null) {
            Row(Modifier.fillMaxWidth()) {
                Metric("Board type", "${b.boardType} rev ${b.revision}", Modifier.weight(1.4f))
                Metric("Age", "${b.ageYears} yrs", Modifier.weight(0.8f))
                Metric("Prev. repairs", "${b.previousRepairs}", Modifier.weight(1f))
            }
            Caption("Last recorded fault: ${faultName(b.lastFault)}")
        }
    }
}

// ---------------------------------------------------------------- 3. part detection
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PartsStep(vm: BenchViewModel) {
    StepCard(3, "Components on the board") {
        KindBadge("AI", "Part detector (YOLOv8, trained by us, on-device)")
        when {
            !vm.detectorAvailable -> Caption("Part detector model is not bundled in this build.")
            vm.photo == null -> Caption("Load a board photo to detect its components.")
            vm.partsBusy -> Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.size(20.dp)); Text("  Detecting parts...")
            }
            vm.parts.isEmpty() -> Caption("No components detected above the confidence threshold.")
            else -> {
                val names = vm.parts.map { it.name }.distinct().sorted()
                Text("${vm.parts.size} components found in ${vm.partsMs} ms",
                    fontWeight = FontWeight.SemiBold, modifier = Modifier.testTag("parts_summary"))
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    vm.partCounts().forEach { (name, n) ->
                        AssistChip(onClick = {}, label = { Text("$name × $n") }, leadingIcon = {
                            Canvas(Modifier.size(10.dp)) { drawCircle(PartColors[names.indexOf(name) % PartColors.size]) }
                        })
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 4. voice note
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun VoiceStep(vm: BenchViewModel, engine: Engine) {
    val context = LocalContext.current
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) vm.toggleRecording() else vm.message = "Microphone permission is needed for voice notes"
    }
    StepCard(4, "Voice note") {
        KindBadge("AI", "Speech-to-text (Vosk, offline)")
        Button(
            onClick = {
                val has = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                        PackageManager.PERMISSION_GRANTED
                if (has || vm.recording) vm.toggleRecording() else permission.launch(Manifest.permission.RECORD_AUDIO)
            },
            colors = if (vm.recording) ButtonDefaults.buttonColors(containerColor = Critical) else ButtonDefaults.buttonColors(),
            modifier = Modifier.testTag("mic"),
        ) {
            Icon(if (vm.recording) Icons.Filled.Stop else Icons.Filled.Mic, null)
            Text(if (vm.recording) "  Stop" else "  Record voice note")
        }
        if (vm.speechStatus.isNotBlank()) Caption(vm.speechStatus)
        OutlinedTextField(
            value = vm.note, onValueChange = { vm.note = it },
            label = { Text("Technician note (editable)") },
            modifier = Modifier.fillMaxWidth().testTag("note"), minLines = 3,
        )
        Caption("Or tap an example:")
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            engine.examplesNotes.forEachIndexed { i, ex ->
                AssistChip(onClick = { vm.note = ex }, label = { Text(ex.take(32) + "…") },
                    modifier = Modifier.testTag("example_$i"))
            }
        }
    }
}

// ---------------------------------------------------------------- 5. diagnosis
@Composable
private fun DiagnosisStep(vm: BenchViewModel, d: com.pcbrepair.assistant.Diagnosis) {
    StepCard(5, "Fault diagnosis") {
        KindBadge("ML", "Fault classifier (TF-IDF + Logistic Regression)")
        Metric("Predicted fault", faultName(d.top), Modifier.testTag("fault"))
        Caption("${(d.confidence * 100).toInt()}% confidence")
        d.classes.indices.sortedByDescending { d.proba[it] }.forEach { i ->
            BarRow(faultName(d.classes[i]), d.proba[i].toFloat(), "${(d.proba[i] * 100).toInt()}%")
        }
        if (d.words.isNotEmpty()) {
            Text("Words that drove this decision: " + d.words.joinToString("  ·  "), fontSize = 13.sp, color = Ink2)
        }
    }
}

// ---------------------------------------------------------------- 6. recommendation
@Composable
private fun RecommendationStep(vm: BenchViewModel, d: com.pcbrepair.assistant.Diagnosis) {
    val r = vm.recommend(d) ?: return
    val b = vm.board() ?: return
    StepCard(6, "Repair recommendation") {
        KindBadge("ML", "Random Forest: BER risk + estimated hours")
        Row(Modifier.fillMaxWidth()) {
            Metric("Recommendation", if (r.ber) "⛔ BER" else "✅ Repairable", Modifier.weight(1.3f).testTag("recommendation"),
                color = if (r.ber) Critical else Ink)
            Metric("BER risk", "${(r.pBer * 100).toInt()}%", Modifier.weight(1f))
            Metric("Est. time", "%.1f h".format(r.hours), Modifier.weight(1f))
        }
        Caption("Inputs: ${b.boardType} · ${faultName(d.top)} · ${b.ageYears} yrs · ${b.previousRepairs} previous repairs. BER = beyond economic repair.")
    }
}
