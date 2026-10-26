package com.pcbrepair.assistant.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

val Blue = Color(0xFF2A78D6)
val BlueDark = Color(0xFF1C5CAB)
val Ink = Color(0xFF0B0B0B)
val Ink2 = Color(0xFF52514E)
val Muted = Color(0xFF898781)
val Track = Color(0xFFE1E0D9)
val Surface = Color(0xFFFCFCFB)
val Good = Color(0xFF0CA30C)
val Critical = Color(0xFFD03B3B)

/** Categorical colours for part classes, fixed order (never cycled by rank). */
val PartColors = listOf(
    Color(0xFF2A78D6), Color(0xFFEB6834), Color(0xFF1BAF7A), Color(0xFFEDA100),
    Color(0xFFE87BA4), Color(0xFF008300), Color(0xFF6250D6), Color(0xFFE34948),
)

val AppColors = lightColorScheme(
    primary = Blue, onPrimary = Color.White, secondary = BlueDark,
    background = Surface, surface = Color.White, onSurface = Ink, onBackground = Ink,
    surfaceVariant = Color(0xFFF1F0EC),
)

@Composable
fun AppTheme(content: @Composable () -> Unit) = MaterialTheme(colorScheme = AppColors, content = content)

/** "AI" (pretrained) vs "ML" (trained by us) tag shown on every step. */
@Composable
fun KindBadge(kind: String, text: String) {
    val (fg, bg) = if (kind == "AI") BlueDark to Color(0xFFE6F0FC) else Color(0xFF8A3C12) to Color(0xFFFDEEE6)
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            kind, color = fg, fontWeight = FontWeight.Bold, fontSize = 12.sp,
            modifier = Modifier.clip(RoundedCornerShape(6.dp)).background(bg).padding(horizontal = 8.dp, vertical = 2.dp),
        )
        Spacer(Modifier.width(8.dp))
        Text(text, fontWeight = FontWeight.SemiBold, fontSize = 14.sp, color = Ink)
    }
}

@Composable
fun StepCard(number: Int?, title: String, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(1.dp),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(if (number != null) "$number · $title" else title, style = MaterialTheme.typography.titleLarge, color = Ink)
            content()
        }
    }
}

@Composable
fun Metric(label: String, value: String, modifier: Modifier = Modifier, color: Color = Ink) {
    Column(modifier) {
        Text(label, fontSize = 12.sp, color = Ink2)
        Text(value, fontSize = 22.sp, color = color, fontWeight = FontWeight.Medium)
    }
}

/** Horizontal bar: label, track, value text. Text stays in ink colours. */
@Composable
fun BarRow(label: String, fraction: Float, valueText: String, color: Color = Blue, labelWidth: Int = 130) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
        Text(label, fontSize = 13.sp, color = Ink2, modifier = Modifier.width(labelWidth.dp))
        Box(Modifier.weight(1f).height(14.dp)) {
            Box(Modifier.fillMaxWidth().height(14.dp).clip(RoundedCornerShape(4.dp)).background(Track.copy(alpha = 0.5f)))
            Box(
                Modifier.fillMaxWidth(fraction.coerceIn(0.005f, 1f)).height(14.dp)
                    .clip(RoundedCornerShape(topEnd = 4.dp, bottomEnd = 4.dp)).background(color)
            )
        }
        Text(valueText, fontSize = 13.sp, color = Ink, modifier = Modifier.width(56.dp).padding(start = 8.dp))
    }
}

@Composable
fun Caption(text: String) = Text(text, fontSize = 12.sp, color = Ink2)
