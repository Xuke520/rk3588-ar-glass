import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gesture import Camera,GestureThread,SharedMemory

if __name__ == "__main__":
    camera=Camera(21)
    w=camera.width
    h=camera.height
    gesture_thread=GestureThread(camera)
    shm=SharedMemory("gesture",size=gesture_thread.max_hands*(21*3*4))
    shm.create()
    gesture_thread.start()
    try:
        while True:
            landmarks=gesture_thread.get_landmarks()
            shm.write_landmarks(landmarks)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    gesture_thread.stop()
    shm.destroy()



