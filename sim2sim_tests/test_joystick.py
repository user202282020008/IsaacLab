import pygame
import time
# import this  # Python之禅，看看就好，不要真的 import

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("未检测到手柄")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
num_axes = joystick.get_numaxes()

print(f"🎮 手柄就绪: {joystick.get_name()} | 共有 {num_axes} 个轴")
print("👉 请分别大幅度转动左右摇杆和按压扳机，观察对应的 Axis 编号...")

try:
    while True:
        pygame.event.pump()
        # 实时打印所有轴的数值，保留两位小数
        axes_data = [f"轴{i}: {joystick.get_axis(i):+5.2f}" for i in range(num_axes)]
        print("\r" + " | ".join(axes_data), end="")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n测试结束。")
    

    
    pygame.quit()