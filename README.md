YOLOv8 및 OpenCV를 활용한 차선 및 차량 감지 시스템

**개요**
이 프로젝트는 YOLOv8 (You Only Look Once) 및 OpenCV를 사용하여 차선 및 차량 감지 시스템을 구축하는 방법을 보여줍니다. 이 시스템은 도로 차선을 감지하고 차량을 식별하며 카메라로부터의 거리를 추정할 수 있습니다. 컴퓨터 비전 기술과 딥러닝 기반 객체 감지를 결합하여 도로 환경에 대한 포괄적인 이해를 제공합니다.

**주요 기능**
**차선 감지:** 에지(Edge) 감지 및 허프 변환(Hough Line Transformation)을 사용하여 도로 차선을 감지합니다.
**차량 감지:** YOLOv8을 사용하여 차량을 식별하고, 차량 주변에 경계 상자를 그립니다.
**거리 추정:** 감지된 차량의 경계 상자 크기를 기반으로 카메라로부터의 거리를 계산합니다.

# 설치

1. 종속성 설치 :
   ```
   pip install opencv-python-headless numpy ultralytics
   ```
2. video.py 파일을 실행한다.
   
# 실행 단계
## 1. 차선 감지 pip
차선 감지 프로세스는 다음 단계로 구성됩니다:

## 1단계: 관심 영역(ROI) 마스킹
이미지의 하단 부분(일반적으로 차선이 보이는 부분)만 처리합니다.
```
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```
## 2단계: 캐니(Canny)를 사용한 에지 감지
이미지를 회색(gray)로 변환하고 캐니에지(Canny Edge)감지를 적용하여 에지를 강조합니다.
```
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cannyed_image = cv2.Canny(gray_image, 100, 200)
```
## 3단계: 허프 선 변환
차선을 나타내는 선을 감지하기 위해 허프 선 변환을 적용합니다.
```
lines = cv2.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)
```
## YOLOv8을 사용한 차량 감지

**Step 1:** Load the YOLOv8 Model
We use a pre-trained YOLOv8 model to detect cars in each frame.
```
from ultralytics import YOLO
model = YOLO('weights/yolov8n.pt')
```
**Step 2:** Draw Bounding Boxes
For each detected car, we draw bounding boxes and display the class name (car) with a confidence score.
```
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = box.conf[0]
    if model.names[cls] == 'car' and conf >= 0.5:
        label = f'{model.names[cls]} {conf:.2f}'
        cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(lane_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
```
3. Distance Estimation
We estimate the distance to each detected car based on the size of the bounding box.
```
def estimate_distance(bbox_width, bbox_height):
    focal_length = 1000  # Example focal length
    known_width = 2.0  # Approximate width of a car (in meters)
    distance = (known_width * focal_length) / bbox_width
    return distance
```
4. Video Processing Pipeline
We combine lane detection, car detection, and distance estimation into a real-time video processing pipeline.
```
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    lane_frame = pipeline(resized_frame)
    results = model(resized_frame)
    for result in results:
        # Draw bounding boxes and estimate distance
    cv2.imshow('Lane and Car Detection', lane_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```
# Results
The system detects lanes and cars in real time, displaying bounding boxes for detected vehicles and estimating their distance from the camera.
