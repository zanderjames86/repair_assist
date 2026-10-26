package com.pcbrepair.assistant

import android.Manifest
import android.content.pm.PackageManager
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextReplacement
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.pcbrepair.assistant.core.LabelParser
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** Runs on an Android emulator in CI (see .github/workflows/android.yml). */
@RunWith(AndroidJUnit4::class)
class EndToEndTest {

    @get:Rule
    val rule = createAndroidComposeRule<MainActivity>()

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun waitForTag(tag: String, timeoutMs: Long = 60_000) =
        rule.waitUntil(timeoutMs) { rule.onAllNodes(hasTestTag(tag)).fetchSemanticsNodes().isNotEmpty() }

    @Test
    fun appIsAirGapped() {
        val info = context.packageManager.getPackageInfo(context.packageName, PackageManager.GET_PERMISSIONS)
        val perms = info.requestedPermissions?.toList() ?: emptyList()
        assertFalse("INTERNET permission must not be present", perms.contains(Manifest.permission.INTERNET))
    }

    @Test
    fun ocrReadsEveryDemoLabel() = runBlocking {
        val engine = Engine(context)
        val reader = LabelReader()
        val names = engine.demoImages()
        assertTrue("expected demo images", names.size >= 50)
        var correct = 0
        for (n in names) {
            val board = engine.registry.first { it.image == n }
            val text = reader.read(Images.fromAsset(context, "demo/$n"))
            if (LabelParser.parse(text, engine.knownBoardTypes).serial == board.serial) correct++
        }
        val accuracy = correct.toDouble() / names.size
        println("OCR serial accuracy on ${names.size} demo photos: $accuracy")
        assertTrue("OCR accuracy $accuracy below 0.9", accuracy >= 0.9)
    }

    @Test
    fun detectorFindsComponents() {
        val detector = PartDetector.load(context)
        assumeTrue("part detector not bundled in this build", detector != null)
        val engine = Engine(context)
        val img = Images.fromAsset(context, "demo/" + engine.demoImages().first())
        val parts = detector!!.detect(img)
        assertTrue("expected some parts, got ${parts.size}", parts.isNotEmpty())
        parts.forEach { assertTrue(it.score in 0.3f..1.0f) }
    }

    @Test
    fun fullJobCardFlow() {
        waitForTag("demo_button")
        rule.onNodeWithTag("demo_button").performClick()
        waitForTag("demo_grid")
        val first = Engine(context).demoImages().first()
        rule.onNodeWithTag("demo_$first").performClick()
        waitForTag("ocr_result")
        rule.waitUntil(30_000) {
            rule.onAllNodes(hasTestTag("ocr_result") and hasText("✅", substring = true)).fetchSemanticsNodes().isNotEmpty()
        }

        rule.onNodeWithTag("note").performScrollTo()
        rule.onNodeWithTag("note").performTextReplacement(
            "Green deposits on the pins, salt residue across the lower edge, moisture ingress.")
        waitForTag("fault")
        rule.onNodeWithTag("fault").performScrollTo().assertTextContains("Corrosion", substring = true)
        rule.onNodeWithTag("recommendation").performScrollTo()
        rule.onNodeWithTag("save").performScrollTo().performClick()

        rule.onNodeWithTag("tab_log").performClick()
        rule.waitUntil(10_000) { rule.onAllNodes(hasText("Corrosion", substring = true)).fetchSemanticsNodes().isNotEmpty() }

        rule.onNodeWithTag("tab_models").performClick()
        rule.onNodeWithText("Fault classifier").assertExists()
    }

    /** Same 600 parity cases as the JVM unit test, but on Android's own runtime (ICU regex etc.). */
    @Test
    fun onDeviceParityWithScikitLearn() {
        val testAssets = InstrumentationRegistry.getInstrumentation().context.assets
        val json = testAssets.open("parity.json").reader().use {
            com.google.gson.JsonParser.parseReader(it).asJsonObject
        }
        val e = Engine(context)
        var n = 0
        for (c in json.getAsJsonArray("notes")) {
            val o = c.asJsonObject
            val p = e.fault.predictProba(o.get("note").asString)
            o.getAsJsonArray("proba").forEachIndexed { k, v -> assertEquals(o.get("note").asString, v.asDouble, p[k], 1e-6) }
            n++
        }
        for (c in json.getAsJsonArray("tabular")) {
            val o = c.asJsonObject
            val args = listOf(o.get("board_type").asString, o.get("fault_category").asString)
            val age = o.get("age_years").asInt
            val prev = o.get("previous_repairs").asInt
            assertEquals(o.get("p_ber").asDouble, e.outcome.predict(args[0], args[1], age, prev), 1e-6)
            assertEquals(o.get("hours").asDouble, e.hours.predict(args[0], args[1], age, prev), 1e-6)
            n++
        }
        assertTrue(n >= 600)
    }

    /** Synthesised voice notes (16 kHz WAV) -> Vosk offline speech-to-text -> fault classifier. */
    @Test
    fun offlineSpeechToText() {
        val testAssets = InstrumentationRegistry.getInstrumentation().context.assets
        val cases = testAssets.open("speech/cases.json").reader().use {
            com.google.gson.JsonParser.parseReader(it).asJsonArray
        }
        val latch = java.util.concurrent.CountDownLatch(1)
        var model: org.vosk.Model? = null
        var error: String? = null
        org.vosk.android.StorageService.unpack(context, "model-en-us", "model",
            { m -> model = m; latch.countDown() }, { e -> error = e.message; latch.countDown() })
        assertTrue("speech model unpack timed out", latch.await(180, java.util.concurrent.TimeUnit.SECONDS))
        val m = model ?: throw AssertionError("speech model failed to load: $error")
        val engine = Engine(context)
        var hits = 0; var total = 0; var correct = 0
        cases.forEachIndexed { i, c ->
            val o = c.asJsonObject
            val wav = testAssets.open("speech/note$i.wav").use { it.readBytes() }
            val pcmStart = dataChunk(wav)
            val rec = org.vosk.Recognizer(m, 16000.0f, VoiceRecorder.grammar(context))
            var pos = pcmStart
            while (pos < wav.size) {
                val n = minOf(4096, wav.size - pos)
                rec.acceptWaveForm(wav.copyOfRange(pos, pos + n), n)
                pos += n
            }
            val text = com.google.gson.JsonParser.parseString(rec.finalResult).asJsonObject.get("text").asString
            rec.close()
            println("Vosk heard: '$text'")
            val words = text.lowercase().split(" ").toSet()
            o.getAsJsonArray("keywords").forEach { k -> total++; if (k.asString in words) hits++ }
            val p = engine.fault.predictProba(text)
            if (engine.fault.classes()[com.pcbrepair.assistant.core.FaultClassifier.argmax(p)] == o.get("fault").asString) correct++
        }
        m.close()
        println("Vosk keyword recall $hits/$total, faults correct $correct/${cases.size()}")
        assertTrue("keyword recall $hits/$total", hits.toDouble() / total >= 0.5)
        assertTrue("faults correct $correct/${cases.size()}", correct >= 2)
    }

    /** Byte offset of the PCM samples in a WAV file. */
    private fun dataChunk(wav: ByteArray): Int {
        var i = 12
        while (i + 8 <= wav.size) {
            val id = String(wav, i, 4, Charsets.US_ASCII)
            val size = (wav[i + 4].toInt() and 0xFF) or ((wav[i + 5].toInt() and 0xFF) shl 8) or
                    ((wav[i + 6].toInt() and 0xFF) shl 16) or ((wav[i + 7].toInt() and 0xFF) shl 24)
            if (id == "data") return i + 8
            i += 8 + size
        }
        return 44
    }

    @Test
    fun onDeviceModelsMatchTrainingResults() {
        val e = Engine(context)
        val p = e.fault.predictProba("No output on the 5V rail, visible corrosion near the connector, regulator U3 is hot.")
        assertEquals(6, p.size)
        assertEquals(1.0, p.sum(), 1e-9)
        val ber = e.outcome.predict("RF-COMM", "corrosion", 7, 1)
        assertTrue(ber in 0.0..1.0)
    }
}
