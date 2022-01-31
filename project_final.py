import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import random

target = random.randrange(-15, 15)

actions = ['up', 'down', 'stop']
seq_length = 30

cur_score = random.randrange(-15, 15)
diff_score = 1

model = load_model('models/model1.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []

while cap.isOpened():

    ret, img = cap.read()  # 프레임 잘 읽으면 ret = true

    if ret is not True:
        continue

    img = cv2.flip(img, 1)  # image flip 0 : updown 1: leftright
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    result = hands.process(img)  # RGB으로 처리
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
            v = v2 - v1  # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])  # matrix of 100 data

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)  # expand datas

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))  # indice (index) of the prediction
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2]:  # if last 3 motions are same
                this_action = action

            if cur_score != target:
                if this_action == 'up':
                    cur_score += diff_score
                elif this_action == 'down':
                    cur_score -= diff_score
                elif this_action == 'stop':
                    cur_score += 0

            cv2.putText(img, f'{this_action.upper()}',
                        org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)

    if cur_score == target:
        stat = 'Right number'
    elif cur_score < target:
        stat = 'Need higher number'
    else:
        stat = 'Need lower number'
    cv2.putText(img, f'{stat}', org=(100, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(255, 255, 255), thickness=2)
    cv2.putText(img, f'{cur_score}', org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 255, 255), thickness=2)
    cv2.putText(img, f'Get this number : {target}', org=(100, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 255, 255), thickness=2)

    cv2.imshow('hands', img)

    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('r'):
        cur_score = random.randrange(-15, 15)
        target = random.randrange(-15, 15)
        cv2.waitKey(3000)
