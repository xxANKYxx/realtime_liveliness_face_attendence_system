import os
from imutils.video import VideoStream
import cv2
import time

def generate_faces(new_path, new_face):
    video_capture = VideoStream(src=0).start()
    count=1
    while True:
        frame = video_capture.read()
        cv2.imshow("recording faces...", frame)
        key_pressed = cv2.waitKey(500) # wait 0.5 second
        filename = new_path + '/' + new_face + str(count) + '.jpg'
        if not(key_pressed & 0xFF == ord('q')): # q=quit
            cv2.imwrite(filename, frame)
            count += 1
        else:
            break

    cv2.destroyAllWindows()
    video_capture.stop()
    print("[LOG] recording done.")
    status=1
    return status

if __name__ == "__main__":
    face_dir = 'faces/'
    new_face = 'ankit'
    new_path = face_dir + new_face

    # if sub-directory with new name does not exist, then create
    cwd = os.getcwd()
    if os.path.exists(new_path):
        print('Sub directory: "', new_path + '" exists in', cwd, '- please remove it first')
    else:
        try:
            os.mkdir(new_path)
            print('Sub directory: "', new_path + '" created')
            print('Generating images of face...', new_face)
            if generate_faces(new_path, new_face):
                print("success")
            else:
                print("failed")
        except FileExistsError:
            print('Sub directory: "', new_path + '" already exist')