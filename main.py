from src.opencv_motion import run as run_opencv


def main():
    print("Практика: анализ видеопотока регистратора")
    print("1 — OpenCV: разность кадров")
    print("2 — YOLOv8n: обнаружение объектов")
    print("3 — Запустить оба метода")
    choice = input("Выберите режим (1/2/3): ").strip()

    if choice == "1":
        run_opencv()
    elif choice == "2":
        from src.yolo_detect import run as run_yolo
        run_yolo()
    elif choice == "3":
        run_opencv()
        from src.yolo_detect import run as run_yolo
        run_yolo()
    else:
        print("Нужно ввести 1, 2 или 3.")


if __name__ == "__main__":
    main()
