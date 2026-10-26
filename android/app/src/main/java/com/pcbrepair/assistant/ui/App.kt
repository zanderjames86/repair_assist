package com.pcbrepair.assistant.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Insights
import androidx.compose.material.icons.filled.ListAlt
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.lifecycle.viewmodel.compose.viewModel
import com.pcbrepair.assistant.BenchViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun App(vm: BenchViewModel = viewModel()) {
    AppTheme {
        var tab by rememberSaveable { mutableIntStateOf(0) }
        val snackbar = remember { SnackbarHostState() }
        LaunchedEffect(vm.message) {
            vm.message?.let { snackbar.showSnackbar(it); vm.message = null }
        }
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("PCB Repair Assistant") },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = BlueDark, titleContentColor = Color.White
                    ),
                )
            },
            snackbarHost = { SnackbarHost(snackbar) },
            bottomBar = {
                NavigationBar {
                    NavigationBarItem(selected = tab == 0, onClick = { tab = 0 },
                        icon = { Icon(Icons.Filled.Build, null) }, label = { Text("Bench") },
                        modifier = Modifier.testTag("tab_bench"))
                    NavigationBarItem(selected = tab == 1, onClick = { tab = 1 },
                        icon = { Icon(Icons.Filled.Insights, null) }, label = { Text("Models") },
                        modifier = Modifier.testTag("tab_models"))
                    NavigationBarItem(selected = tab == 2, onClick = { tab = 2 },
                        icon = { Icon(Icons.Filled.ListAlt, null) }, label = { Text("Log") },
                        modifier = Modifier.testTag("tab_log"))
                }
            },
        ) { pad ->
            Box(Modifier.padding(pad).fillMaxSize()) {
                val engine = vm.engine
                when {
                    vm.loadError != null -> Text("Failed to load models: ${vm.loadError}", Modifier.align(Alignment.Center))
                    engine == null -> Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator()
                        Text("Loading on-device models...")
                    }
                    tab == 0 -> BenchScreen(vm, engine)
                    tab == 1 -> ModelsScreen(vm, engine)
                    else -> LogScreen(vm)
                }
            }
        }
    }
}
