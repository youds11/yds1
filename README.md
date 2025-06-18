## YOLOv8 및 OpenCV를 활용한 차선 및 차량 감지 시스템

## 개요
이 프로젝트는 YOLOv8 (You Only Look Once) 및 OpenCV를 사용하여 차선 및 차량 감지 시스템을 구축하는 방법을 보여줍니다. 이 시스템은 도로 차선을 감지하고 차량을 식별하며 카메라로부터의 거리를 추정할 수 있습니다. 컴퓨터 비전 기술과 딥러닝 기반 객체 감지를 결합하여 도로 환경에 대한 포괄적인 이해를 제공합니다.

## 주요 기능
**차선 감지:** 에지(Edge) 감지 및 허프 변환(Hough Line Transformation)을 사용하여 도로 차선을 감지합니다.

**차량 감지:** YOLOv8을 사용하여 차량을 식별하고, 차량 주변에 경계 상자를 그립니다.

**거리 추정:** 감지된 차량의 경계 상자 크기를 기반으로 카메라로부터의 거리를 계산합니다.

# 설치

1. 종속성 설치 :
   ```
   pip install opencv-python-headless numpy ultralytics
   ```
   
2. **video.py** 파일을 실행한다.
   
# 실행 단계

## 1. 차선 감지 
차선 감지 프로세스는 다음 단계로 구성됩니다:

**1단계:** 관심 영역(ROI) 마스킹
이미지의 하단 부분(일반적으로 차선이 보이는 부분)만 처리합니다.
```
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

**2단계:** 캐니(Canny)를 사용한 에지 감지
이미지를 회색(gray)로 변환하고 캐니에지(Canny Edge)감지를 적용하여 에지를 강조합니다.
```
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cannyed_image = cv2.Canny(gray_image, 100, 200)
```

**3단계:** 허프 선 변환
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

**1단계:** YOLOv8 모델 로드
사전 훈련된 YOLOv8 모델을 사용하여 각 프레임에서 차량을 감지합니다.
```
from ultralytics import YOLO
model = YOLO('weights/yolov8n.pt')
```

**2단계:** 경계 상자 그리기
감지된 각 차량에 대해 경계 상자를 그리고 신뢰도 점수와 함께 클래스 이름(car)을 표시합니다.
```
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = box.conf[0]
    if model.names[cls] == 'car' and conf >= 0.5:
        label = f'{model.names[cls]} {conf:.2f}'
        cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(lane_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
```

## 3. 거리측정
감지된 각 차량까지의 거리를 바운딩박스의 크기를 기반으로 측정합니다.
```
def estimate_distance(bbox_width, bbox_height):
    focal_length = 1000  # Example focal length
    known_width = 2.0  # Approximate width of a car (in meters)
    distance = (known_width * focal_length) / bbox_width
    return distance
```

## 4. 영상처리 
차선 감지, 차량 감지 및 거리 추정을 실시간 비디오 처리 파이프라인으로 결합한다.
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
```

# 결과
이 시스템은 차선과 차량을 실시간으로 감지하며, 감지된 차량에 대한 바운딩박스로 표시하고 카메라로부터의 거리를 측정합니다. 
하지만 거리 추정에 사용된 초점 거리(focal length) 1000 및 차량 크기(known_width) 2.0은 임의의 값으로 설정된 것입니다. 따라서 실제 환경에서의 거리 계산은 약간의 오차가 있을 수 있으며 더 정확한 측정을 위해서는 본인의 카메라에 맞는 초점 거리(f)값을 찾아야합니니다.
카메라 캘리브레이션을 통해 카메라를 보정하는 것이 중요합니다.
