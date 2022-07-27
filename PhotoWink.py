import cv2
import mediapipe as mp
import time
import pyautogui
import os

mp_face_mesh = mp.solutions.face_mesh
# eyes_indices2 = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158,
#                 157, 173, 263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387,
#                 386, 385, 384, 398]  # all landmarks of eyes
eyes_indices = [263, 387, 386, 385, 362, 380, 374, 373, 33, 160, 159, 158, 133, 153, 145, 144]
save_path = input("请将保存照片的文件夹拖入控制台，显示文件夹路径后按回车开始拍照\n")
wCam, hCam = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # 设置帧宽度
cap.set(4, hCam)  # 设置帧高度
pTime = 0
timeToTakePhoto = 0
start_flag = 0
sleep_flag = 0
sleep = 0
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
     min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.resize(image, dsize=(wCam, hCam))
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 提取需要的眼部特征点
        eyes_landmarks = {}
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for ix in eyes_indices:
                    x = face_landmarks.landmark[ix].x * wCam
                    y = face_landmarks.landmark[ix].y * hCam
                    eyes_landmarks[ix] = (x, y)

        # 计算EAR值
        if len(eyes_landmarks) == 16:
            EAR = (abs(eyes_landmarks[158][1]-eyes_landmarks[153][1]) + abs(eyes_landmarks[159][1]-eyes_landmarks[145][1]) + \
                   abs(eyes_landmarks[160][1]-eyes_landmarks[144][1])) / 3 / (abs(eyes_landmarks[133][0]-eyes_landmarks[33][0]))

        # 特征点及帧率显示
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_to_write = image.copy()
        img_to_write = cv2.flip(img_to_write, 1)
        if eyes_landmarks:
            for key in eyes_landmarks:
                lm_x, lm_y = eyes_landmarks[key]
                image = cv2.circle(image, (int(lm_x), int(lm_y)), 3, (0, 255, 255), cv2.FILLED)
        image = cv2.flip(image, 1)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # 根据EAR值确定右眼开始闭眼的时间
        if EAR < 0.11:  # 右眼闭合且未处于拍照倒计时状态
            if start_flag == 0:
                # 闭眼时间大于1秒则开始拍照计时
                if sleep_flag == 2:
                    timeToTakePhoto = time.time()
                    start_flag = 1
                    sleep_flag = 0
                elif sleep_flag == 1 and sleep > 0:
                    curr = time.time()
                    if curr - sleep > 2:
                        sleep_flag = 2
                elif sleep_flag == 0:
                    sleep = time.time()
                    sleep_flag = 1
        else:
            sleep_flag = 0
        
        if start_flag == 1:
            curr_timeToTakePhoto = time.time()
            timeDiff = curr_timeToTakePhoto - timeToTakePhoto
            if timeDiff > 5:
                fileList = os.listdir(save_path)
                i = 0
                while True:
                    photoName = f"Photo{i}.jpg"
                    if photoName in fileList:
                        i += 1
                    else:
                        break
                save_path2 = os.path.join(save_path, photoName)
                cv2.imwrite(save_path2, img_to_write)
                res = pyautogui.confirm(text='是否继续拍照？', title='拍照成功', buttons=['继续', '退出'])  # OK和Cancel按钮的消息弹窗
                if res == '退出':
                    break
                start_flag = 0
            else:
                cv2.putText(image, str(int(4-timeDiff)), (wCam//2, 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)

        cv2.imshow('PhotoWink', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
cap.release()