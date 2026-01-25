# import tkinter as tk
#
# def main():
#     # 其他初始化代码
#     print("Found 7 age indices and 7 sph_r indices")
#     print("Target age index: 17, Target sph index: None")
#
#     # 模拟多轮运行
#     for simulation in range(2):
#         for step in range(3):
#             predicted_value = -1.9015 + (step * 0.3025)
#             new_age = 11.50 + (step * 0.50)
#             print(f"Simulation {simulation}, Step {step}:")
#             print(f"Predicted value: {predicted_value:.4f}")
#             print(f"New age: {new_age:.2f}")
#
#     # Tkinter调用（仅作为示例）
#     root = tk.Tk()
#     root.title("Test Window")
#     label = tk.Label(root, text="Hello, Tkinter!")
#     label.pack(pady=20)
#     root.mainloop()
#
# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print(f"预测过程中出现错误: {e}")



import matplotlib.font_manager

# 获取所有可用字体
font_list = sorted({f.name for f in matplotlib.font_manager.fontManager.ttflist})

for font in font_list:
    print(font)