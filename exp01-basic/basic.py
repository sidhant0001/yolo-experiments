from ultralytics import YOLO

model = YOLO("yolo26n.pt")

#version1
# results = model("https://ultralytics.com/images/bus.jpg", save=True)
# print(results[0].boxes.cls)         # class IDs of what it found

#version2
# result = model(
#     "src-images/car.jpg",
#     save=True,
#     project="output-inference")

#version3
result = model(
    "src-images/car.jpg",
    save=True,
    project=r"C:\SID\Yolo\1-getting-started\exp01-basic\output-inference")