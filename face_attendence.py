import os
import face_recognition
from datetime import date
import cv2
from collections import defaultdict
from imutils.video import VideoStream
from eye_cond import *
from tqdm import tqdm


def init():
    face_detector=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    left_eye_detector=cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    right_eye_detector=cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    open_eyes_detector=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    datasets='faces'
    print("webcam opening")
    # video_cap=VideoStream(src=0).start()
    video_cap=VideoStream(src=0).start()


    model=load_model()

    print('collecting images')
    images=[]
    for directory,_,files in tqdm(os.walk(datasets)):
        for file in files:
            if file.endswith("jpg"):
                images.append(os.path.join(directory,file))

    return (model,face_detector,open_eyes_detector,left_eye_detector,right_eye_detector,video_cap,images)

def process_and_encode(images):
    encoding_known=[]
    known_names=[]

    for image_path in tqdm(images):
        image=cv2.imread(image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        boxes=face_recognition.face_locations(image,model='hog')
        encoding =face_recognition.face_encodings(image,boxes)
        name=image_path.split(os.path.sep)[-2]
        if len(encoding)>0:
            encoding_known.append(encoding[0])
            known_names.append(name)
    return {"encoding":encoding_known,"names":known_names}

def eyeBlinking(history,maxFrames):
    # history==A tring containg the history of eye status(open or closed)
    # maxFrames==max num of successive frame where eye are closed

    for i in range(maxFrames):
        pattern='1'+'0'*(i+1)+'1'
        if pattern in history:
            return True
        return False


def doAttendence(name):
    with open('Attendence.csv','r+') as f:
        attendencelist=f.readlines()
        nameOfStudent=[]
        today = date.today()
        d1 = today.strftime("%d/%m/%Y")
        for line in attendencelist:
            entry=line.split(',')
            nameOfStudent.append(entry[0])
        if name not in nameOfStudent:
            info=f'{name},{d1}'
            f.writelines(f'\n{name},{d1}')



def detect_display(model,video_cap,face_detector,open_eyes_detector,left_eye_detector,right_eye_detector,data,eyes_detected):
    frame=video_cap.read()
    frame=cv2.resize(frame,(0,0),fx=0.6,fy=0.6)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #detect face
    faces = face_detector.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(50, 50),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        encoding=face_recognition.face_encodings(rgb,[(y, x+w, y+h, x)])[0]
        matches=face_recognition.compare_faces(data["encoding"],encoding)
        name="unknown"
        if True in matches:
            matchedIndex=[i for (i, b) in enumerate(matches) if b]
            counts={}
            for i in matchedIndex:
                name=data["names"][i]
                counts[name]=counts.get(name,0)+1
            name=max(counts,key=counts.get)

        face=frame[y:y+h,x:x+w]
        gray_face=gray[y:y+h,x:x+w]
        eye=[]
        # Eyes detection
        # check first if eyes are open (with glasses taking into account)
        open_eyes_glasses = open_eyes_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # if open_eyes_glasses detect eyes then they are open
        if len(open_eyes_glasses) == 2:
            eyes_detected[name] += '1'
            for (ex, ey, ew, eh) in open_eyes_glasses:
                cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # otherwise try detecting eyes using left and right_eye_detector
        # which can detect open and closed eyes
        else:
            # separate the face into left and right sides
            left_face = frame[y:y + h, x + int(w / 2):x + w]
            left_face_gray = gray[y:y + h, x + int(w / 2):x + w]

            right_face = frame[y:y + h, x:x + int(w / 2)]
            right_face_gray = gray[y:y + h, x:x + int(w / 2)]

            # Detect the left eye
            left_eye = left_eye_detector.detectMultiScale(left_face_gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

            # Detect the right eye
            right_eye = right_eye_detector.detectMultiScale(right_face_gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

            eye_status = '1'  # we suppose the eyes are open

            # For each eye check  the eye is closed.
            # If one is closed we conclude the eyes are closed
            for (ex, ey, ew, eh) in right_eye:
                color = (0, 255, 0)
                pred = predict(right_face[ey:ey + eh, ex:ex + ew],model)
                if pred == 'closed':
                    eye_status = '0'
                    color = (0, 0, 255)
                cv2.rectangle(right_face, (ex, ey), (ex + ew, ey + eh), color, 2)
            for (ex, ey, ew, eh) in left_eye:
                color = (0, 255, 0)
                pred = predict(left_face[ey:ey + eh, ex:ex + ew],model)
                if pred == 'closed':
                    eye_status = '0'
                    color = (0, 0, 255)
                cv2.rectangle(left_face, (ex, ey), (ex + ew, ey + eh), color, 2)
            eyes_detected[name] += eye_status

            if eyeBlinking(eyes_detected[name], 3):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Displays name
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                doAttendence(name)
    return frame




if __name__=="__main__":
    (model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, video_cap, images)=init()
    data=process_and_encode(images)
    eyes_detected=defaultdict(str)
    while True:
        frame=detect_display(model,video_cap,face_detector,open_eyes_detector,left_eye_detector,right_eye_detector,data,eyes_detected)
        cv2.imshow("Attendence system",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    video_cap.stop()