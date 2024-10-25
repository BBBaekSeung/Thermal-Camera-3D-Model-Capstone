from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#얼굴사진 이미지 업로드
new_image1_path = 'st1.jpg'
new_image2_path = 'st2.jpg'

new_image1 = Image.open(new_image1_path)
new_image2 = Image.open(new_image2_path)

#그레이스케일로 변환
new_image1_gray = new_image1.convert('L')
new_image2_gray = new_image2.convert('L')

#뎁스 맵 계산
new_depth_map = np.abs(np.array(new_image1_gray, dtype=np.float32) - np.array(new_image2_gray, dtype=np.float32))

new_depth_map_normalized = (new_depth_map - new_depth_map.min()) / (new_depth_map.max() - new_depth_map.min()) * 255
new_depth_map_normalized = new_depth_map_normalized.astype(np.uint8)

depth_map_image_path = 'face_depth_map.jpg'
depth_map_image = Image.fromarray(new_depth_map_normalized)
depth_map_image.save(depth_map_image_path)

#뎁스 맵을 그레이스케일로 변환
plt.figure(figsize=(6, 6))
plt.title('New Depth Map - Grayscale')
plt.imshow(new_depth_map_normalized, cmap='gray')
plt.axis('off')
plt.show()

depth_map_image_path
