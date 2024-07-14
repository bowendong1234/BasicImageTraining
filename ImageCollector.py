import cv2
import os
import time

def capture_images(class_name, num_images, train_or_val_or_test):
    cap = cv2.VideoCapture(0)
    count = 0
    save_path = os.path.join('data', train_or_val_or_test, class_name)
    os.makedirs(save_path, exist_ok=True)
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        
        # Save frame
        file_name = os.path.join(save_path, f'{class_name}_{count}.jpg')
        cv2.imwrite(file_name, frame)
        count += 1
        time.sleep(0.5)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# capture_images('water_bottle', 100, 'train')
# capture_images('phone', 100, 'train')

# capture_images('water_bottle', 25, 'val')
# capture_images('phone', 25, 'val')

# capture_images('phone', 10, 'test')
# capture_images('phone', 10, 'test')
