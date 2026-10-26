package com.pcbrepair.assistant

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.pcbrepair.assistant.core.YoloDecoder
import kotlinx.coroutines.tasks.await
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.vosk.android.StorageService
import java.io.InputStream
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.roundToInt

// ---------------------------------------------------------------- images
object Images {
    private const val MAX_SIDE = 1600

    fun fromAsset(context: Context, path: String): Bitmap =
        context.assets.open(path).use { decode(it, null) }

    fun fromUri(context: Context, uri: Uri): Bitmap {
        val rotation = context.contentResolver.openInputStream(uri)?.use { exifRotation(it) } ?: 0
        val bmp = context.contentResolver.openInputStream(uri)!!.use { decode(it, null) }
        return if (rotation == 0) bmp else Bitmap.createBitmap(
            bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(rotation.toFloat()) }, true
        )
    }

    private fun exifRotation(stream: InputStream): Int =
        when (ExifInterface(stream).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90
            ExifInterface.ORIENTATION_ROTATE_180 -> 180
            ExifInterface.ORIENTATION_ROTATE_270 -> 270
            else -> 0
        }

    private fun decode(stream: InputStream, opts: BitmapFactory.Options?): Bitmap {
        val raw = BitmapFactory.decodeStream(stream, null, opts)
            ?: throw IllegalArgumentException("Not an image")
        val scale = MAX_SIDE.toFloat() / max(raw.width, raw.height)
        if (scale >= 1f) return raw
        return Bitmap.createScaledBitmap(raw, (raw.width * scale).roundToInt(), (raw.height * scale).roundToInt(), true)
    }
}

// ---------------------------------------------------------------- AI 1: OCR
/** ML Kit text recognition with the model bundled in the APK. */
class LabelReader {
    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    suspend fun read(bitmap: Bitmap): String =
        recognizer.process(InputImage.fromBitmap(bitmap, 0)).await().text

    fun close() = recognizer.close()
}

// ---------------------------------------------------------------- AI 3: part detector
data class Part(val name: String, val score: Float, val box: YoloDecoder.Detection)

/** YOLOv8 part detector (ONNX Runtime, CPU). Returns null if no model is bundled. */
class PartDetector private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    val names: List<String>,
    private val size: Int,
) {
    companion object {
        fun load(context: Context): PartDetector? = runCatching {
            val meta = context.assets.open("models/parts.json").reader().use {
                Gson().fromJson(it, JsonObject::class.java)
            }
            val names = meta.getAsJsonArray("names").map { it.asString }
            val size = meta.get("imgsz")?.asInt ?: 640
            val bytes = context.assets.open("models/parts.onnx").use { it.readBytes() }
            val env = OrtEnvironment.getEnvironment()
            val session = env.createSession(bytes, OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) })
            PartDetector(env, session, names, size)
        }.getOrNull()
    }

    fun detect(bitmap: Bitmap, scoreThreshold: Float = 0.30f, iou: Float = 0.45f): List<Part> {
        val lb = YoloDecoder.Letterbox(bitmap.width, bitmap.height, size)
        val resized = Bitmap.createScaledBitmap(bitmap, lb.newW, lb.newH, true)
        val px = IntArray(lb.newW * lb.newH)
        resized.getPixels(px, 0, lb.newW, 0, 0, lb.newW, lb.newH)
        val input = lb.toTensor(px)
        val shape = longArrayOf(1, 3, size.toLong(), size.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { tensor ->
            session.run(mapOf(session.inputNames.first() to tensor)).use { result ->
                val out = result[0] as OnnxTensor
                val dims = out.info.shape            // [1, 4 + nc, N]
                val buf = out.floatBuffer
                val flat = FloatArray(buf.remaining()).also { buf.get(it) }
                val nc = dims[1].toInt() - 4
                val n = dims[2].toInt()
                return YoloDecoder.decode(flat, nc, n, lb, bitmap.width, bitmap.height, scoreThreshold, iou)
                    .map { Part(names.getOrElse(it.classId) { "class ${it.classId}" }, it.score, it) }
            }
        }
    }
}

// ---------------------------------------------------------------- AI 2: speech-to-text
/** Offline speech recognition with Vosk; the English model ships inside the APK. */
class VoiceRecorder(private val context: Context) {
    private var model: Model? = null
    private var service: SpeechService? = null

    fun prepare(onReady: () -> Unit, onError: (String) -> Unit) {
        if (model != null) { onReady(); return }
        StorageService.unpack(context, "model-en-us", "model",
            { m -> model = m; onReady() },
            { e -> onError("Speech model missing: ${e.message}") })
    }

    /** Streams text while the technician talks; [onUpdate] gets the full transcript so far. */
    fun start(onUpdate: (String) -> Unit, onError: (String) -> Unit) {
        val m = model ?: return onError("Speech model not loaded")
        val rec = Recognizer(m, 16000.0f)
        var committed = ""
        fun join(a: String, b: String) = listOf(a, b).filter { it.isNotBlank() }.joinToString(" ")
        service = SpeechService(rec, 16000.0f).also {
            it.startListening(object : RecognitionListener {
                override fun onPartialResult(hypothesis: String?) {
                    onUpdate(join(committed, field(hypothesis, "partial")))
                }
                override fun onResult(hypothesis: String?) {
                    committed = join(committed, field(hypothesis, "text")); onUpdate(committed)
                }
                override fun onFinalResult(hypothesis: String?) {
                    committed = join(committed, field(hypothesis, "text")); onUpdate(committed)
                }
                override fun onError(exception: Exception?) { onError(exception?.message ?: "error") }
                override fun onTimeout() {}
            })
        }
    }

    fun stop() { service?.stop(); service = null }

    fun release() { service?.shutdown(); service = null; model?.close(); model = null }

    private fun field(json: String?, key: String): String =
        runCatching { JSONObject(json ?: "{}").optString(key, "") }.getOrDefault("")
}
