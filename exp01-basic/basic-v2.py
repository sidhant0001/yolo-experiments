from os.path import exists

from ultralytics import YOLO

model = YOLO("yolo26n.pt")

result = model(
    "src-images/car.jpg",
    save=True,
    exist_ok = True,
    save_txt=True,
    save_conf=True,
    name="city-street",
    save_crop=True,
    conf=0.5,
    project=r"C:\SID\Yolo\1-getting-started\exp01-basic\output-inference")