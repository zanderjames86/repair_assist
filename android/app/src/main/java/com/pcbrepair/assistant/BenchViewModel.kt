package com.pcbrepair.assistant

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.gson.Gson
import com.pcbrepair.assistant.core.FaultClassifier
import com.pcbrepair.assistant.core.LabelParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class Diagnosis(
    val classes: List<String>,
    val proba: List<Double>,
    val top: String,
    val confidence: Double,
    val words: List<String>,
)

data class Recommendation(val pBer: Double, val hours: Double) {
    val ber get() = pBer >= 0.5
}

data class JobCard(
    val time: String, val serial: String, val boardType: String, val note: String,
    val fault: String, val confidence: Double, val recommendation: String,
    val berRisk: Double, val estHours: Double, val parts: String,
)

class BenchViewModel(app: Application) : AndroidViewModel(app) {

    var engine by mutableStateOf<Engine?>(null); private set
    var loadError by mutableStateOf<String?>(null); private set
    private var detector: PartDetector? = null
    var detectorAvailable by mutableStateOf(false); private set
    private val labelReader = LabelReader()
    val voice = VoiceRecorder(app)

    // Step 1: photo + OCR
    var photo by mutableStateOf<Bitmap?>(null); private set
    var photoName by mutableStateOf(""); private set
    var ocrBusy by mutableStateOf(false); private set
    var ocrText by mutableStateOf(""); private set
    var label by mutableStateOf<LabelParser.Label?>(null); private set

    // Step 2: board
    var serial by mutableStateOf<String?>(null)

    // Step 3: parts
    var partsBusy by mutableStateOf(false); private set
    var parts by mutableStateOf<List<Part>>(emptyList()); private set
    var partsMs by mutableStateOf(0L); private set

    // Step 4: voice note
    var note by mutableStateOf("")
    var recording by mutableStateOf(false); private set
    var speechStatus by mutableStateOf("")

    val log = mutableStateListOf<JobCard>()
    private val logFile = File(app.filesDir, "repair_log.json")
    var message by mutableStateOf<String?>(null)

    init {
        viewModelScope.launch {
            try {
                val e = withContext(Dispatchers.Default) { Engine(getApplication()) }
                detector = withContext(Dispatchers.Default) { PartDetector.load(getApplication()) }
                detectorAvailable = detector != null
                engine = e
                loadLog()
            } catch (t: Throwable) {
                loadError = t.message ?: t.toString()
            }
        }
    }

    // ------------------------------------------------------------ photo
    fun loadDemo(name: String) = loadPhoto(name) { Images.fromAsset(getApplication(), "demo/$name") }

    fun loadUri(uri: Uri, name: String) = loadPhoto(name) { Images.fromUri(getApplication(), uri) }

    private fun loadPhoto(name: String, decode: () -> Bitmap) {
        viewModelScope.launch {
            ocrBusy = true
            parts = emptyList()
            try {
                val bmp = withContext(Dispatchers.IO) { decode() }
                photo = bmp
                photoName = name
                val text = labelReader.read(bmp)
                ocrText = text
                val parsed = LabelParser.parse(text, engine?.knownBoardTypes ?: emptyList())
                label = parsed
                if (engine?.board(parsed.serial) != null) serial = parsed.serial
                runDetector(bmp)
            } catch (t: Throwable) {
                message = "Could not read image: ${t.message}"
            } finally {
                ocrBusy = false
            }
        }
    }

    private fun runDetector(bmp: Bitmap) {
        val d = detector ?: return
        viewModelScope.launch {
            partsBusy = true
            val t0 = System.currentTimeMillis()
            parts = withContext(Dispatchers.Default) { runCatching { d.detect(bmp) }.getOrDefault(emptyList()) }
            partsMs = System.currentTimeMillis() - t0
            partsBusy = false
        }
    }

    // ------------------------------------------------------------ voice
    fun toggleRecording() {
        if (recording) {
            voice.stop(); recording = false; speechStatus = ""
            return
        }
        speechStatus = "Loading speech model..."
        voice.prepare(onReady = {
            speechStatus = "Listening... speak now"
            recording = true
            voice.start(
                onUpdate = { text -> note = text },
                onError = { e -> speechStatus = "Speech error: $e"; recording = false },
            )
        }, onError = { e -> speechStatus = e })
    }

    // ------------------------------------------------------------ ML
    fun board(): Board? = engine?.board(serial)

    fun diagnose(): Diagnosis? {
        val e = engine ?: return null
        if (note.isBlank()) return null
        val p = e.fault.predictProba(note)
        val k = FaultClassifier.argmax(p)
        val classes = e.fault.classes().toList()
        return Diagnosis(
            classes = classes, proba = p.toList(), top = classes[k], confidence = p[k],
            words = e.fault.explain(note, k, 6).map { it.term },
        )
    }

    fun recommend(d: Diagnosis?): Recommendation? {
        val e = engine ?: return null
        val b = board() ?: return null
        d ?: return null
        return Recommendation(
            pBer = e.outcome.predict(b.boardType, d.top, b.ageYears, b.previousRepairs),
            hours = e.hours.predict(b.boardType, d.top, b.ageYears, b.previousRepairs),
        )
    }

    fun partCounts(): List<Pair<String, Int>> =
        parts.groupingBy { it.name }.eachCount().toList().sortedByDescending { it.second }

    fun save() {
        val b = board() ?: return
        val d = diagnose() ?: return
        val r = recommend(d) ?: return
        log.add(0, JobCard(
            time = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date()),
            serial = b.serial, boardType = b.boardType, note = note,
            fault = faultName(d.top), confidence = round2(d.confidence),
            recommendation = if (r.ber) "BER" else "Repairable",
            berRisk = round2(r.pBer), estHours = Math.round(r.hours * 10) / 10.0,
            parts = partCounts().joinToString("; ") { "${it.first} x${it.second}" },
        ))
        persistLog()
        message = "Saved job card for S/N ${b.serial}"
    }

    fun clearLog() { log.clear(); persistLog() }

    private fun round2(v: Double) = Math.round(v * 100) / 100.0

    private fun loadLog() {
        runCatching {
            if (logFile.exists()) log.addAll(Gson().fromJson(logFile.readText(), Array<JobCard>::class.java))
        }
    }

    private fun persistLog() {
        runCatching { logFile.writeText(Gson().toJson(log.toList())) }
    }

    fun csv(): String {
        fun q(s: String) = "\"" + s.replace("\"", "\"\"") + "\""
        val header = "time,serial,board_type,note,fault,confidence,recommendation,ber_risk,est_hours,parts"
        return (listOf(header) + log.map {
            listOf(q(it.time), q(it.serial), q(it.boardType), q(it.note), q(it.fault), it.confidence.toString(),
                q(it.recommendation), it.berRisk.toString(), it.estHours.toString(), q(it.parts)).joinToString(",")
        }).joinToString("\n")
    }

    fun exportCsv(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val ok = runCatching {
                getApplication<Application>().contentResolver.openOutputStream(uri)!!.use {
                    it.write(csv().toByteArray())
                }
            }.isSuccess
            message = if (ok) "Exported ${log.size} job cards" else "Export failed"
        }
    }

    override fun onCleared() {
        voice.release()
        labelReader.close()
    }
}
