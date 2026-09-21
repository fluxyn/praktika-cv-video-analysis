from collections import Counter
from pathlib import Path
import csv

from ultralytics import YOLO

INPUT_VIDEO = Path(__file__).resolve().parent.parent / "IMG_6381.MP4"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.25
OUTPUT_CSV = OUTPUT_DIR / "yolo_frame_results.csv"
OUTPUT_STATS = OUTPUT_DIR / "yolo_stats.txt"


def run():
    print("Загрузка YOLOv8n...")
    model = YOLO(MODEL_NAME)

    print("Запуск обнаружения объектов...")
    results = model.predict(
        source=str(INPUT_VIDEO),
        save=True,
        conf=CONFIDENCE,
        verbose=True,
    )

    counts = Counter()
    confidences = []
    frame_rows = []

    for frame_number, result in enumerate(results):
        objects = 0
        details = []

        if result.boxes is not None:
            for cls, conf in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                class_name = model.names[int(cls)]
                confidence = float(conf)
                counts[class_name] += 1
                confidences.append(confidence)
                objects += 1
                details.append(f"{class_name}:{confidence:.4f}")

        frame_rows.append((frame_number, objects, "; ".join(details)))

    total = sum(counts.values())
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    max_objects = max((row[1] for row in frame_rows), default=0)
    avg_objects = sum(row[1] for row in frame_rows) / len(frame_rows) if frame_rows else 0

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "objects", "details"])
        writer.writerows(frame_rows)

    lines = [
        "Результаты YOLOv8n",
        f"Видео: {INPUT_VIDEO.name}",
        f"Модель: {MODEL_NAME}",
        f"Порог confidence: {CONFIDENCE}",
        "",
        "Количество обнаружений по классам:",
    ]
    lines.extend(f"{name}: {count}" for name, count in counts.most_common())
    lines.extend([
        "",
        f"Всего обнаружений: {total}",
        f"Средняя уверенность: {avg_conf:.4f}",
        f"Максимальное количество объектов в одном кадре: {max_objects}",
        f"Среднее количество объектов в кадре: {avg_objects:.2f}",
    ])
    OUTPUT_STATS.write_text("\n".join(lines), encoding="utf-8")

    print("\n==============================")
    print("РЕЗУЛЬТАТЫ YOLO")
    print("==============================")
    for name, count in counts.most_common():
        print(f"{name}: {count}")
    print(f"\nВсего обнаружений: {total}")
    print(f"Средняя уверенность: {avg_conf:.4f}")
    print(f"Максимальное количество объектов в одном кадре: {max_objects}")
    print(f"Среднее количество объектов в кадре: {avg_objects:.2f}")
    print(f"\nТаблица: {OUTPUT_CSV}")
    print(f"Статистика: {OUTPUT_STATS}")
    print("Аннотированное видео YOLO сохраняется Ultralytics в runs/detect/predict.")


if __name__ == "__main__":
    run()
