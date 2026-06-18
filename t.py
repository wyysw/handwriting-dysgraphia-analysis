import cv2

# 读取图片
img = cv2.imread("sym_helper_mask.png")

# 颜色反转
result = 255 - img

# 保存结果
cv2.imwrite("sym_helper.png", result)

print("处理完成")
