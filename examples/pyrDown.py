import cv2
import torch
import fastcv

img = cv2.imread("artifacts/test.jpg")
img_tensor = torch.from_numpy(img).cuda()
pyrDown_tensor = fastcv.pyrDown(img_tensor)

pyrDown_image = pyrDown_tensor.cpu().numpy()
cv2.imwrite("output_pyrDown.jpg", pyrDown_image)

print("saved pyrDowned image.")