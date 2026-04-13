from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# results = model("https://ultralytics.com/images/bus.jpg", save=True)
# print(results[0].boxes.cls)         # class IDs of what it found


result = model(
    "src-images/car.jpg"
)