package com.example.tensorflow2;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.tensorflow2.ml.Flowers;

import com.example.tensorflow2.ml.Heroes;
import com.example.tensorflow2.ml.Prueba;
import com.example.tensorflow2.ml.Tomatodo;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity
        extends AppCompatActivity
        implements OnSuccessListener<Text>,
        OnFailureListener {
    public static int REQUEST_CAMERA = 111;
    public static int REQUEST_GALLERY = 222;

    Bitmap mSelectedImage;
    ImageView mImageView;
    TextView txtResults;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mImageView = findViewById(R.id.image_view);
        txtResults = findViewById(R.id.txtresults);
    }
    public void abrirGaleria (View view){
        Intent i = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(i, REQUEST_GALLERY);
    }
    public void onCameraButtonClick(View view) {
        launchCamera();
    }

    private void launchCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE_SECURE);
        startActivityForResult(cameraIntent, REQUEST_CAMERA);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            handleImageCapture(requestCode, data);
        }
    }

    private void handleImageCapture(int requestCode, Intent data) {
        try {
            mSelectedImage = requestCode == REQUEST_CAMERA ? (Bitmap) data.getExtras().get("data")
                    : MediaStore.Images.Media.getBitmap(getContentResolver(), data.getData());
            displayCapturedImage();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void displayCapturedImage() {
        mImageView.setImageBitmap(mSelectedImage);
    }

    public void OCRfx(View v) {
        InputImage image = InputImage.fromBitmap(mSelectedImage, 0);
        TextRecognizer recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        recognizer.process(image)
                .addOnSuccessListener(this)
                .addOnFailureListener(this);
    }

    @Override
    public void onFailure(@NonNull Exception e) {

    }

    @Override
    public void onSuccess(Text text) {
        List<Text.TextBlock> blocks = text.getTextBlocks();
        String resultados="";
        if (blocks.size() == 0) {
            resultados = "No hay Texto";
        }else{
            for (int i = 0; i < blocks.size(); i++) {
                List<Text.Line> lines = blocks.get(i).getLines();
                for (int j = 0; j < lines.size(); j++) {
                    List<Text.Element> elements = lines.get(j).getElements();
                    for (int k = 0; k < elements.size(); k++) {
                        resultados = resultados + elements.get(k).getText() + " ";
                    }
                }
                resultados=resultados + "\n";
            }
            resultados=resultados + "\n";
        }
        txtResults.setText(resultados);
    }
    public void Rostrosfx(View v) {
        BitmapDrawable drawable = (BitmapDrawable) mImageView.getDrawable();
        Bitmap bitmap = drawable.getBitmap().copy(Bitmap.Config.ARGB_8888,true);
        Canvas canvas = new Canvas(bitmap);
        Paint paint = new Paint();
        paint.setColor(Color.BLUE);
        paint.setStrokeWidth(15);
        paint.setStyle(Paint.Style.STROKE);

        InputImage image = InputImage.fromBitmap(mSelectedImage, 0);
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                        .build();
        FaceDetector detector = FaceDetection.getClient(options);
        detector.process(image)
                .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
                        if (faces.size() == 0) {
                            txtResults.setText("No Hay rostros");
                        }else{
                            txtResults.setText("Hay " + faces.size() + " rostro(s)");

                            for (Face rostro: faces) {
                                canvas.drawRect(rostro.getBoundingBox(), paint);
                            }

                        }
                    }
                })
                .addOnFailureListener(this);

        mImageView.setImageBitmap(bitmap);
    }

    public void PersonalizedModel(View v) {
        try {
            // Cargar el modelo de héroes
            Heroes model = Heroes.newInstance(getApplicationContext());

            // Aplicar transformaciones a la imagen antes de escalarla
            Bitmap imagen_preprocesada = preprocessImage(mSelectedImage);

            // Escalar la imagen preprocesada a 224x224 píxeles
            Bitmap imagen_escalada = Bitmap.createScaledBitmap(imagen_preprocesada, 224, 224, true);

            // Convertir la imagen escalada en un objeto TensorImage
            TensorImage imagen = new TensorImage(DataType.FLOAT32);
            imagen.load(imagen_escalada);

            // Crear un TensorBuffer para la entrada del modelo
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(imagen.getBuffer());

            // Procesar la imagen con el modelo
            Heroes.Outputs outputs = model.process(inputFeature0);

            // Obtener la salida del modelo como un TensorBuffer
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Etiquetas de los héroes
            String[] etiquetas = { "Iron Man", "Thor", "Hulk", "BlackWidow", "Hawkeye" };

            // Obtener las probabilidades de clasificación
            float[] probabilidades = outputFeature0.getFloatArray();

            // Crear una lista de etiquetas y probabilidades
            List<String> resultados = new ArrayList<>();
            for (int i = 0; i < etiquetas.length; i++) {
                resultados.add(etiquetas[i] + ": " + probabilidades[i] * 100);
            }

            // Mostrar los resultados en tu interfaz de usuario (por ejemplo, en un TextView o RecyclerView)
            String resultadoFinal = TextUtils.join("\n", resultados);
            txtResults.setText("Resultados:\n" + resultadoFinal);

            // Cerrar el modelo después de usarlo
            model.close();
        } catch (IOException e) {
            // Manejar errores en caso de problemas al cargar o procesar el modelo
            txtResults.setText("Error al procesar Modelo");
        }
    }

    private Bitmap preprocessImage(Bitmap image) {
        // Aplicar transformaciones a la imagen antes de escalarla
        // Por ejemplo, puedes aplicar aumento de contraste, normalización y recorte aquí.
        // Devuelve la imagen preprocesada.
        // Aquí tienes un ejemplo simple de escala:
        int targetSize = 224;
        Bitmap scaledImage = Bitmap.createScaledBitmap(image, targetSize, targetSize, true);
        return scaledImage;
    }



    class CategoryComparator implements java.util.Comparator<Category> {
        @Override
        public int compare(Category a, Category b) {
            return (int)(b.getScore()*100) - (int)(a.getScore()*100);
        }
    }
}