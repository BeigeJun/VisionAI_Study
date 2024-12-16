from pynput.keyboard import Key, Controller, Listener
import time

keyboard = Controller()
running = True

def on_press(key):
    global running
    if key == Key.esc:
        running = False
        return False

listener = Listener(on_press=on_press)
listener.start()

while running:
    # 스페이스바 누르기
    keyboard.press(Key.space)
    keyboard.release(Key.space)
    time.sleep(1)  # 1초 대기

    # 백스페이스 누르기
    keyboard.press(Key.backspace)
    keyboard.release(Key.backspace)
    time.sleep(1)  # 1초 대기

print("프로그램이 종료되었습니다.")
