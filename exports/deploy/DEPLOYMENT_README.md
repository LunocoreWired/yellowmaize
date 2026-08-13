# MAIze Android Deployment Guide
Generated: 2026-07-31 01:54:42

## Files in this package

| File | Size | Purpose |
|---|---|---|
| `bouncer_model.tflite` | See report | Gate: maize vs non-maize |
| `student_model.tflite` | See report | Disease detection (3 heads) |
| `model_metadata.json`  | —          | Complete input/output specs |
| `deployment_report.csv`| —          | Size + latency summary |

Status: Bouncer ✓ ready | Student ✓ ready

---

## Android Studio Integration

### 1. Add TFLite dependency (build.gradle)
```groovy
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'  // optional
}
```

### 2. Copy model files
```
app/src/main/assets/
    bouncer_model.tflite
    student_model.tflite
```

### 3. Preprocessing (Java/Kotlin)
```kotlin
// Letterbox resize to 224×224 (longest side → 224, pad shorter side with zeros)
val imageProcessor = ImageProcessor.Builder()
    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
    .add(NormalizeOp(
        floatArrayOf(0.485f, 0.456f, 0.406f),  // mean (ImageNet)
        floatArrayOf(0.229f, 0.224f, 0.225f)    // std  (ImageNet)
    ))
    .build()

// Apply EXIF orientation first
val bitmap = ExifInterface(imagePath).let {
    BitmapFactory.decodeFile(imagePath)
        .rotateBitmap(it.getAttributeInt(ExifInterface.TAG_ORIENTATION, 1))
}

val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
```

### 4. Heuristic pre-filter (before Bouncer)
```kotlin
// Fast green coverage check — reject non-plant images immediately
fun isLikelyMaize(bitmap: Bitmap): Boolean {
    val hsv = FloatArray(3)
    var greenPixels = 0
    val total = bitmap.width * bitmap.height
    for (x in 0 until bitmap.width step 4) {
        for (y in 0 until bitmap.height step 4) {
            Color.colorToHSV(bitmap.getPixel(x, y), hsv)
            val h = hsv[0]; val s = hsv[1] * 255; val v = hsv[2] * 255
            if (h in 35f..75f && s > 50 && v > 40) greenPixels++
        }
    }
    return (greenPixels.toFloat() / (total / 16)) > 0.15f
}
```

### 5. Run Bouncer
```kotlin
val bouncerInterpreter = Interpreter(
    FileUtil.loadMappedFile(context, "bouncer_model.tflite"))

val bouncerOutput = Array(1) { FloatArray(1) }
bouncerInterpreter.run(tensorImage.buffer, bouncerOutput)

val sigmoid = 1f / (1f + Math.exp(-bouncerOutput[0][0].toDouble())).toFloat()
if (sigmoid < 0.7f) {
    showMessage("Not detected as a maize leaf")
    return
}
```

### 6. Run Student
```kotlin
val studentInterpreter = Interpreter(
    FileUtil.loadMappedFile(context, "student_model.tflite"))

// Output buffers — TFLite outputs in NHWC layout
val segOutput = Array(1) { Array(224) { Array(224) { FloatArray(2) } } }  // [1,H,W,2]
val clsOutput = Array(1) { FloatArray(3) }                                     // [1,3]
val sevOutput = Array(1) { FloatArray(1) }                                     // [1,1]

val outputs = mapOf(0 to segOutput, 1 to clsOutput, 2 to sevOutput)
studentInterpreter.runForMultipleInputsOutputs(
    arrayOf(tensorImage.buffer), outputs)
```

### 7. Post-processing
```kotlin
// Classification
val classProbs = softmax(clsOutput[0])
val classIndex = classProbs.indexOfMax()
val className  = arrayOf("HEALTHY", "MSV", "MLN")[classIndex]
val confidence = classProbs[classIndex] * 100f

// Severity
val severityPct = sevOutput[0][0] * 100f

// Segmentation — NHWC layout: segOutput[0][y][x][channel]
// channel 0 = leaf silhouette, channel 1 = symptom mask
val silhouette = Array(224) { y -> FloatArray(224) { x -> segOutput[0][y][x][0] } }
val symptoms   = Array(224) { y -> FloatArray(224) { x -> segOutput[0][y][x][1] } }
// Apply sigmoid and threshold at 0.5 to get binary masks
// Draw green contour overlay from silhouette mask
// Draw symptom region overlay from symptoms mask

// UI labels
displayResult(
    className, confidence, severityPct,
    silhouetteMask = binarize(sigmoid(silhouette), 0.5f),
    symptomMask    = binarize(sigmoid(symptoms),   0.5f)
)
```

### 8. Display outputs
- **Green contour** from `silhouette` channel → label: "Symptom boundary"
- **Amber heatmap** from Grad-CAM++ on classification output → label: "Diagnostic attention"
- **Class badge**: HEALTHY / MSV / MLN with confidence %
- **Severity gauge**: 0–100% bar

---

## Important notes

1. **Input normalization** — must use ImageNet mean/std exactly as above. Wrong normalization produces random outputs with no error.
2. **Channel order** — TFLite uses NHWC (channels last): shape `[1, 224, 224, 3]`. The ONNX→TF→TFLite conversion handles the PyTorch NCHW→NHWC transpose automatically. Pass your Android `TensorImage` directly — do NOT manually transpose to NCHW.
3. **Sigmoid vs softmax** — segmentation and severity outputs need sigmoid; classification needs softmax. Applying the wrong function produces incorrect results silently.
4. **Severity disclaimer** — severity % is learned from HSV-derived pseudo-labels, not expert agronomic ratings. Display as an estimate, not a diagnosis.
5. **Bouncer threshold** — the value `0.7` was empirically selected on the validation split. Do not hardcode a different value.

---

## Full model_metadata.json
See `model_metadata.json` for complete technical specifications including all
input/output tensor shapes, normalization parameters, and post-processing steps.
