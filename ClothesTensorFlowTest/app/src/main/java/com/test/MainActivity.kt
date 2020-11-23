package com.test

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.test.ml.ConvertedModel
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "CameraXBasic"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        recognize_button.setOnClickListener { recognize() }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(
            requestCode: Int, permissions: Array<String>, grantResults:
            IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build()
                    .also {
                        val surfaceProvider = camera_preview.surfaceProvider
                        it.setSurfaceProvider(surfaceProvider)
                    }
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun recognize() {
        camera_preview.bitmap
                ?.toGrayScale()
                ?.scale(28, 28)
                ?.toNormalizedByteBuffer()
                ?.let { byteBuffer ->
                    val t = getBitmap(byteBuffer, 28, 28)

                    val model = ConvertedModel.newInstance(this)

                    val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 28, 28), DataType.FLOAT32)
                    inputFeature.loadBuffer(byteBuffer)

                    val outputFeature = model.process(inputFeature).outputFeature0AsTensorBuffer

                    val tensorLabel = TensorLabel(arrayListOf(
                            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
                    ), outputFeature)

                    tensorLabel.mapWithFloatValue.maxByOrNull { it.value }

                    info_container.text = tensorLabel.mapWithFloatValue.maxByOrNull { it.value }?.key + "\n" + tensorLabel.mapWithFloatValue.toString()
                    model.close()
                }
    }

    private fun Bitmap.toGrayScale(): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
        canvas.drawBitmap(this, 0f, 0f, paint)
        return bitmap
    }

    private fun Bitmap.scale(width: Int, height: Int) =
            Bitmap.createScaledBitmap(this, width, height, false)

    private fun Bitmap.toNormalizedByteBuffer(): ByteBuffer? {
        val imageData = ByteBuffer.allocateDirect(4 * width * height)
        imageData.order(ByteOrder.nativeOrder())

        val pixels = IntArray(width * height)
        getPixels(pixels, 0, width, 0, 0, width, height)
        for (pixel in pixels) {
            val value = (Color.red(pixel).toFloat() +
                    Color.blue(pixel).toFloat() +
                    Color.green(pixel).toFloat()) / 3f / 255f
            imageData.putFloat(value)
        }
        return imageData
    }

    private fun getBitmap(buffer: Buffer, width: Int, height: Int): Bitmap? {
        buffer.rewind()
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)
        return bitmap
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
}