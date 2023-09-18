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

import com.example.tensorflow2.ml.Exam;
import com.example.tensorflow2.ml.Flowers;

import com.example.tensorflow2.ml.Heroes;
import com.example.tensorflow2.ml.Lab;
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
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;
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

    public ByteBuffer convertirImagenATensorBuffer(Bitmap mSelectedImage){

        Bitmap imagen = Bitmap.createScaledBitmap(mSelectedImage, 224, 224, true);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[224 * 224];
        imagen.getPixels(intValues, 0, imagen.getWidth(), 0, 0, imagen.getWidth(), imagen.getHeight());

        int pixel = 0;

        for(int i = 0; i <  imagen.getHeight(); i ++){
            for(int j = 0; j < imagen.getWidth(); j++){
                int val = intValues[pixel++]; // RGB
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
            }
        }
        return  byteBuffer;
    }
    public String obtenerEtiquetayProbabilidad(String[] etiquetas, float[] probabilidades){

        float valorMayor=Float.MIN_VALUE;
        int pos=-1;
        for (int i = 0; i < probabilidades.length; i++) {
            if (probabilidades[i] > valorMayor) {
                valorMayor = probabilidades[i];
                pos = i;
            }
        }

        return "Predicción: " + etiquetas[pos] + ", Probabilidad: " + (new DecimalFormat("0.00").format(probabilidades[pos] * 100)) + "%";

    }

    public void PersonalizedModel(View v){
        try {

            //Definir Estiquetas de acuerdo a su archivo "labels.txt" generado por la Plataforma de creación del Modelo
            String[] etiquetas = {"LABORATORIO INDUSTRIAL","POLIDEPORTIVO","CENTRO MEDICO"};

            Exam model = Exam.newInstance(getApplicationContext());
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(convertirImagenATensorBuffer(mSelectedImage));

            Exam.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            txtResults.setText(obtenerEtiquetayProbabilidad(etiquetas, outputFeature0.getFloatArray()));

            model.close();
        } catch (Exception e) {
            txtResults.setText(e.getMessage());
        }
    }
    public String obtenerEtiquetayProbabilidad2(String[] etiquetas, float[] probabilidades) {
        StringBuilder resultado = new StringBuilder();

        for (int i = 0; i < etiquetas.length; i++) {
            resultado.append(etiquetas[i]).append(": ").append(probabilidades[i] * 100).append("%\n");
        }

        return resultado.toString();
    }

    public void PersonalizedModel2(View v){
        try {

            //Definir Estiquetas de acuerdo a su archivo "labels.txt" generado por la Plataforma de creación del Modelo
            String[] etiquetas = {"LABORATORIO INDUSTRIAL","POLIDEPORTIVO","CENTRO MEDICO"};

            Exam model = Exam.newInstance(getApplicationContext());
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(convertirImagenATensorBuffer(mSelectedImage));

            Exam.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            txtResults.setText(obtenerEtiquetayProbabilidad2(etiquetas, outputFeature0.getFloatArray()));

            model.close();
        } catch (Exception e) {
            txtResults.setText(e.getMessage());
        }
    }

    class CategoryComparator implements java.util.Comparator<Category> {
        @Override
        public int compare(Category a, Category b) {
            return (int)(b.getScore()*100) - (int)(a.getScore()*100);
        }
    }
}