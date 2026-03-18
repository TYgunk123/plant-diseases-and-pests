import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
import tempfile
import shutil
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def preprocess_image(image_path, input_shape, quantized=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 對於非 ASCII 路徑，在 Windows 上直接 cv2.imread 會失敗。先試一般讀取，再試通過 imdecode，最後試 Pillow。
    img = cv2.imread(image_path)
    if img is None:
        # 方案1：使用 OpenCV 二進位解碼，這可繞過 OpenCV path 編碼問題。
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            img_arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        except Exception:
            img = None

    if img is None and PIL_AVAILABLE:
        try:
            with Image.open(image_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img = np.array(pil_img)[:, :, ::-1]  # RGB->BGR
        except Exception as e:
            raise ValueError(f"Cannot read image: {image_path} ({e})")

    if img is None:
        raise ValueError(
            f"Cannot read image: {image_path}. "
            "請確認路徑/檔案是否存在以及是否可讀。"
        )

    height, width = input_shape[1], input_shape[2]
    img = cv2.resize(img, (width, height))

    if quantized:
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)


def infer_tflite(model_path, image_path, labels, use_quantized=None):
    model_path = os.path.abspath(model_path)

    def make_interpreter(path):
        try:
            return tf.lite.Interpreter(model_path=path)
        except ValueError as e:
            if "Could not open" in str(e) and any(ord(ch) > 127 for ch in path):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(path)[1]) as tmp:
                    shutil.copy2(path, tmp.name)
                    tmp_path = tmp.name
                try:
                    return tf.lite.Interpreter(model_path=tmp_path)
                finally:
                    os.remove(tmp_path)
            raise

    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]

    if use_quantized is None:
        quantized = input_details[0]["dtype"] in (np.uint8, np.int8)
    else:
        quantized = use_quantized

    img = preprocess_image(image_path, input_shape, quantized=quantized)

    if quantized and input_details[0].get("quantization") and input_details[0]["quantization"] != (0.0, 0):
        scale, zero_point = input_details[0]["quantization"]
        img = img / scale + zero_point
        img = img.astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    pred_index = int(np.argmax(output_data, axis=1)[0])
    confidence = float(output_data[0][pred_index])
    pred_label = labels[pred_index] if 0 <= pred_index < len(labels) else str(pred_index)

    return pred_label, confidence, output_data


def collect_images(image_path):
    image_path = os.path.abspath(image_path)

    if os.path.isfile(image_path):
        return [image_path]

    if os.path.isdir(image_path):
        image_list = []
        for root, _, files in os.walk(image_path):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                    image_list.append(os.path.join(root, name))
        image_list.sort()
        return image_list

    raise FileNotFoundError(f"圖像路徑不存在: {image_path}")


def main():
    parser = argparse.ArgumentParser(description="一次測試兩個 TFLite 模型的預測結果")
    parser.add_argument("--model1", default=r"C:\Users\ASUS\Desktop\Plant diseases and pests\訓練資料\MobileNetV2.tfliteQuant", help="第一個模型檔案路徑")
    parser.add_argument("--model2", default=r"C:\Users\ASUS\Desktop\Plant diseases and pests\訓練資料\plant_model.tflite", help="第二個模型檔案路徑")
    parser.add_argument("--image", default=r"C:\Users\ASUS\Desktop\Plant diseases and pests\訓練資料\PlantDataset", help="測試圖像路徑(單檔或資料夾)")
    parser.add_argument("--labels", default=r"C:\Users\ASUS\Desktop\Plant diseases and pests\訓練資料\Labels.txt", help="labels 文字檔路徑")
    parser.add_argument("--quantized", action="store_true", help="如果模型是量化模型，使用此旗標")
    parser.add_argument("--output", default=None, help="輸出 CSV 檔案路徑 (可選)")

    args = parser.parse_args()

    labels = load_labels(args.labels)
    images = collect_images(args.image)

    if len(images) == 0:
        print(f"找不到圖像檔: {args.image}")
        return

    results = []
    for idx, model_path in enumerate([args.model1, args.model2], start=1):
        if not os.path.exists(model_path):
            print(f"模型不存在: {model_path}")
            continue

        print(f"\n=== 開始模型 {idx} 推論: {model_path} (總圖像 {len(images)}) ===")

        for image_path in images:
            try:
                label, conf, raw = infer_tflite(model_path, image_path, labels, use_quantized=args.quantized)
            except Exception as e:
                print(f"  [失敗] {image_path} -> {e}")
                continue

            print(f"  {os.path.relpath(image_path)} -> {label} ({conf:.5f})")
            results.append({
                "model": os.path.basename(model_path),
                "image": image_path,
                "pred_label": label,
                "confidence": conf,
            })

    if args.output:
        import csv

        out_path = os.path.abspath(args.output)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "image", "pred_label", "confidence"])
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"\n已輸出結果 CSV: {out_path}")


if __name__ == "__main__":
    main()
