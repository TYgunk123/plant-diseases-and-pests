package com.example.plant_diseases_and_pests_app;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.Manifest;
import android.content.pm.PackageManager;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    Button btnCamera, btnGallery, btnDetect;
    ImageView imagePreview;
    TextView resultText;

    static final int CAMERA_REQUEST = 1;
    static final int GALLERY_REQUEST = 2;

    Bitmap selectedImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 隱藏頂部欄
        if(getSupportActionBar() != null){
            getSupportActionBar().hide();
        }

        btnCamera = findViewById(R.id.btnCamera);
        btnGallery = findViewById(R.id.btnGallery);
        btnDetect = findViewById(R.id.btnDetect);
        imagePreview = findViewById(R.id.imagePreview);
        resultText = findViewById(R.id.resultText);

        // 按鈕事件
        btnCamera.setOnClickListener(v -> openCamera());
        btnGallery.setOnClickListener(v -> openGallery());
        btnDetect.setOnClickListener(v -> detectDisease());
    }

    // --- 開啟相機（按下才請求權限） ---
    void openCamera() {
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED){

            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.CAMERA},
                    100
            );

        } else {
            Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(cameraIntent, CAMERA_REQUEST);
        }
    }

    // --- 開啟相簿（按下才請求權限） ---
    void openGallery() {
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                != PackageManager.PERMISSION_GRANTED){

            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.READ_MEDIA_IMAGES},
                    101
            );

        } else {
            Intent galleryIntent = new Intent(Intent.ACTION_PICK,
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(galleryIntent, GALLERY_REQUEST);
        }
    }

    // --- 權限回傳處理 ---
    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions,
                                           int[] grantResults) {

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if(requestCode == 100 && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED){

            openCamera(); // 同意權限後開啟相機

        } else if(requestCode == 101 && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED){

            openGallery(); // 同意權限後開啟相簿
        }
    }

    // --- 取得回傳的圖片 ---
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        super.onActivityResult(requestCode, resultCode, data);

        if(resultCode == RESULT_OK){

            if(requestCode == CAMERA_REQUEST){
                Bitmap photo = (Bitmap) data.getExtras().get("data");
                selectedImage = photo;
                imagePreview.setImageBitmap(photo);
            }

            if(requestCode == GALLERY_REQUEST){
                try{
                    selectedImage = MediaStore.Images.Media.getBitmap(
                            this.getContentResolver(),
                            data.getData()
                    );
                    imagePreview.setImageBitmap(selectedImage);
                } catch (Exception e){
                    e.printStackTrace();
                }
            }
        }
    }

    // --- 模擬 AI 辨識 ---
    void detectDisease(){
        if(selectedImage == null){
            resultText.setText("Please select an image first");
            return;
        }

        // 模擬辨識結果
        String result = "Leaf Blight\nConfidence: 92%";
        resultText.setText(result);
    }
}