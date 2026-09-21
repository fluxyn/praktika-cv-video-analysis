import cv2
from pathlib import Path

INPUT_VIDEO = Path(__file__).resolve().parent.parent / "IMG_6381.MP4"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_VIDEO = OUTPUT_DIR / "opencv_result.mp4"
STATS_FILE = OUTPUT_DIR / "opencv_stats.txt"


def run():
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Не удалось создать выходной MP4-файл.")

    ret, previous = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise RuntimeError("Не удалось прочитать первый кадр.")

    changed_fractions = []
    processed = 1

    while True:
        ret, current = cap.read()
        if not ret:
            break

        prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, threshold = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)

        changed_fraction = float((threshold > 0).mean() * 100.0)
        changed_fractions.append(changed_fraction)

        contours, _ = cv2.findContours(
            threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = current.copy()
        boxes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                boxes += 1

        cv2.putText(
            result,
            f"Frame: {processed}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        out.write(result)
        previous = current
        processed += 1

    cap.release()
    out.release()

    mean_changed = sum(changed_fractions) / len(changed_fractions) if changed_fractions else 0
    max_changed = max(changed_fractions) if changed_fractions else 0

    STATS_FILE.write_text(
        "Результаты OpenCV — разность кадров\n"
        f"Видео: {INPUT_VIDEO.name}\n"
        f"Разрешение: {width} x {height}\n"
        f"FPS: {fps:.2f}\n"
        f"Кадров: {frame_count}\n"
        f"Средняя доля изменившихся пикселей: {mean_changed:.2f}%\n"
        f"Максимальная доля изменившихся пикселей: {max_changed:.2f}%\n",
        encoding="utf-8",
    )

    print(f"OpenCV готов: {OUTPUT_VIDEO}")
    print(f"Статистика: {STATS_FILE}")


if __name__ == "__main__":
    run()
