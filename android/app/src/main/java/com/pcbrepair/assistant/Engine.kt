package com.pcbrepair.assistant

import android.content.Context
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.annotations.SerializedName
import com.pcbrepair.assistant.core.FaultClassifier
import com.pcbrepair.assistant.core.RandomForest
import java.io.InputStreamReader

/** Board history record (bundled registry, looked up after OCR). */
data class Board(
    val serial: String,
    @SerializedName("board_type") val boardType: String,
    val revision: String,
    @SerializedName("age_years") val ageYears: Int,
    @SerializedName("previous_repairs") val previousRepairs: Int,
    @SerializedName("last_fault") val lastFault: String,
    val image: String? = null,
    val source: String? = null,
)

val FAULT_NAMES = mapOf(
    "short_circuit" to "Short circuit",
    "open_circuit" to "Open circuit",
    "component_failure" to "Component failure",
    "corrosion" to "Corrosion",
    "connector_fault" to "Connector fault",
    "power_supply" to "Power supply",
)

fun faultName(key: String) = FAULT_NAMES[key] ?: key

/** Loads every bundled model once. Nothing here touches the network. */
class Engine(private val context: Context) {
    private val gson = Gson()

    private fun reader(path: String) = InputStreamReader(context.assets.open(path), Charsets.UTF_8)

    val fault: FaultClassifier = reader("models/fault_lr.json").use { FaultClassifier(it) }
    val outcome: RandomForest = reader("models/outcome_rf.json").use { RandomForest(it) }
    val hours: RandomForest = reader("models/hours_rf.json").use { RandomForest(it) }
    val metrics: JsonObject = reader("models/metrics.json").use { gson.fromJson(it, JsonObject::class.java) }
    val registry: List<Board> = reader("models/registry.json").use {
        gson.fromJson(it, Array<Board>::class.java).toList()
    }
    val knownBoardTypes: List<String> = registry.map { it.boardType }.distinct().sorted()

    val partMetrics: JsonObject? = runCatching {
        reader("models/parts.json").use { gson.fromJson(it, JsonObject::class.java) }
    }.getOrNull()

    fun board(serial: String?): Board? = registry.firstOrNull { it.serial == serial }

    fun demoImages(): List<String> =
        (context.assets.list("demo") ?: emptyArray()).filter { it.endsWith(".jpg") }.sorted()

    val examplesNotes = listOf(
        "No output on the 5V rail, visible corrosion near the connector, regulator U3 is hot.",
        "Dead short on the 3.3V rail, capacitor C12 shorted to ground, will replace and retest.",
        "Connector J2 pins bent and loose contact, cable intermittent at the plug.",
        "Green deposits on the pins, salt residue across the lower edge, moisture ingress.",
    )
}
