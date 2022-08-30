import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from tensorflow.keras.models import load_model


max_num_hands = 1
classes = [
    "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ",
    "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ",
    "ㅑ", "ㅓ", "ㅔ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ"
]

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
model = load_model('handtrain(400(87.9)).h5')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = np.full(shape=(512,512,3),fill_value=255,dtype=np.uint8)
    #img = Image.fromarray(img)
    result = hands.process(img)
    

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #img = Image.fromarray(img)
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)

            result = model.predict([data]).squeeze()
            idx = np.argmax(result)
            
            img = Image.fromarray(img)

            draw = ImageDraw.Draw(img)
            font = ImageFont. truetype("fonts/gulim.ttc",100)
            org=(300,50)
            text=classes[idx]
            draw.text(org,text,font=font,fill=(0,0,0))
            img=np.array(img)

            #cv2.putText(img, text=classes[idx], org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Sign Language Recognition', img)
    if cv2.waitKey(1) == ord('q'):
        break