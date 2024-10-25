import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 온도 데이터 불러오기
file_path = '열화상카메라 온도 데이터.csv'
thermography_data = pd.read_csv(file_path)

# DataFrame을 numpy 배열로 변환
temperature_data = thermography_data.values.astype(float)

# 이미지 파일 경로를 올바르게 지정하세요
depth_map_path = "depthmap_cut.png"
thermal_image_path = "tmp_cut.png"

# 깊이 맵 이미지 불러오기
depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
if depth_map is None:
    raise FileNotFoundError(f"깊이 맵 이미지 파일을 찾을 수 없습니다: {depth_map_path}")

# 열화상 이미지 불러오기
thermal_image = cv2.imread(thermal_image_path)
if thermal_image is None:
    raise FileNotFoundError(f"열화상 이미지 파일을 찾을 수 없습니다: {thermal_image_path}")

# 열화상 이미지를 올바른 포맷으로 변환 (BGR에서 RGB로)
thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)

# 깊이 맵과 열화상 이미지의 크기를 동일하게 조정
h, w = depth_map.shape
thermal_image_resized = cv2.resize(thermal_image, (w, h))

# 얼굴 부분을 검출하기 위한 임계값 설정
_, depth_thresh = cv2.threshold(depth_map, 50, 255, cv2.THRESH_BINARY_INV)
_, thermal_thresh = cv2.threshold(cv2.cvtColor(thermal_image_resized, cv2.COLOR_RGB2GRAY), 200, 255, cv2.THRESH_BINARY)

# 깊이 맵의 얼굴 중심 좌표 계산
depth_moments = cv2.moments(depth_thresh)
depth_cx = int(depth_moments['m10'] / depth_moments['m00'])
depth_cy = int(depth_moments['m01'] / depth_moments['m00'])

# 열화상 이미지의 얼굴 중심 좌표 계산
thermal_moments = cv2.moments(thermal_thresh)
thermal_cx = int(thermal_moments['m10'] / thermal_moments['m00'])
thermal_cy = int(thermal_moments['m01'] / thermal_moments['m00'])

# 이미지 정렬을 위한 이동 벡터 계산
dx = depth_cx - thermal_cx
dy = depth_cy - thermal_cy

# 열화상 이미지를 깊이 맵과 정렬
M = np.float32([[1, 0, dx], [0, 1, dy]])
thermal_image_aligned = cv2.warpAffine(thermal_image_resized, M, (w, h))

# (x, y) 좌표 그리드 생성
x = np.linspace(0, w - 1, w)
y = np.linspace(0, h - 1, h)
x, y = np.meshgrid(x, y)

# 깊이 맵을 더 나은 시각화를 위해 정규화
z = depth_map.astype(np.float32)
z = cv2.normalize(z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# 정렬된 열화상 색상을 깊이 맵에 매핑
facecolors = thermal_image_aligned / 255.0

# 3D 플롯 생성 및 열화상 색상 적용
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 정렬된 열화상 이미지로 표면 플롯
surf = ax.plot_surface(x, y, z, facecolors=facecolors, edgecolor='none')

# 컬러바 추가
mappable = plt.cm.ScalarMappable(cmap='inferno')
mappable.set_array(temperature_data)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Temperature (°C)')

ax.set_title('3D Model with Aligned Thermal Colors')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Depth')

plt.show()
