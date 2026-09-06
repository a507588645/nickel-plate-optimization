import cv2

def is_dark_light_clear(img, threshold1=100, threshold2=200, threshold3=500):
    img = cv2.imread(img) # 调试用，如果图片直接从相机而来，就不用这句。
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average_brightness = cv2.mean(gray)[0]
    imageVar = cv2.Laplacian(gray, cv2.CV_64F).var()
    retvalue=0
    if imageVar > threshold3:
        print('clear')
        retvalue= 0
    else:
        print('blur')
        retvalue= 3
        return retvalue
    if average_brightness < threshold1:
        print('dark')
        retvalue=2
    elif average_brightness > threshold2:
        print('light')
        retvalue=1
    else:
        print('normal')
        retvalue=0
    return retvalue

def is_exist_full(img):
     #TODO

    print('lacking datasets')