import pandas as pd
import matplotlib.pyplot as plt

#열화상 데이터 파일 업로드
file_path = '열화상카메라 온도 데이터.csv'

thermography_data = pd.read_csv(file_path)

#DataFrame을 numpy 배열로 변환
temperature_data = thermography_data.values.astype(float)

#열화상 이미지 생성
plt.figure(figsize=(10, 8))
plt.imshow(temperature_data, cmap='inferno')
plt.axis('off')  # 축 제거

image_path_new_no_grid_no_colorbar = './thermal_image_new_no_grid_no_colorbar.png'
plt.savefig(image_path_new_no_grid_no_colorbar, bbox_inches='tight', pad_inches=0)
plt.show()
