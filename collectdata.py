
import os
import cv2
cap=cv2.VideoCapture(0)
import os

directory = 'Image'
for letter in "ABC":
    path = os.path.join(directory, letter)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

while True:
    _,frame=cap.read()
    count = {
             'a': len(os.listdir(directory+"/A")),
             'b': len(os.listdir(directory+"/B")),
             'c': len(os.listdir(directory+"/C")),
    }
          
    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame,(0,40),(300,400),(255,255,255),2)
    cv2.imshow("data",frame)
    cv2.imshow("ROI",frame[40:400,0:300])
    frame=frame[40:400,0:300]
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('a'):
        save_path = os.path.join(directory, 'A', f"{count['a']}.png")
        cv2.imwrite(save_path, frame)
        print(f"Saving A: {save_path}")
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'B/'+str(count['b'])+'.png',frame)
        cv2.imwrite(save_path, frame)
        print(f"Saving B: {save_path}")
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'C/'+str(count['c'])+'.png',frame)
        cv2.imwrite(save_path, frame)
        print(f"Saving C: {save_path}")

    if interrupt & 0xFF == 27:  # 27 is the ASCII code for ESC key
        print("Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()