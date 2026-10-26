package com.pcbrepair.assistant.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.pcbrepair.assistant.BenchViewModel

@Composable
fun LogScreen(vm: BenchViewModel) {
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { uri ->
        if (uri != null) vm.exportCsv(uri)
    }
    Column(Modifier.fillMaxSize().padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { export.launch("repair_log.csv") }, enabled = vm.log.isNotEmpty()) { Text("Export CSV") }
            OutlinedButton(onClick = { vm.clearLog() }, enabled = vm.log.isNotEmpty()) { Text("Clear") }
        }
        Caption("${vm.log.size} job cards, stored on this device only. Export saves a CSV to the tablet's storage or a USB drive.")
        if (vm.log.isEmpty()) Caption("No job cards yet. Complete a job on the Bench tab.")
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.testTag("log_list")) {
            items(vm.log) { j ->
                Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color.White)) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("S/N ${j.serial} · ${j.boardType}", fontWeight = FontWeight.SemiBold)
                        Text("${j.fault} (${(j.confidence * 100).toInt()}%) → ${j.recommendation}, " +
                                "BER risk ${(j.berRisk * 100).toInt()}%, ${j.estHours} h",
                            color = if (j.recommendation == "BER") Critical else Ink)
                        Caption(j.note)
                        if (j.parts.isNotBlank()) Caption("Parts: ${j.parts}")
                        Caption(j.time)
                    }
                }
            }
        }
    }
}
