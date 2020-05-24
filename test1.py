
import cv2
import numpy as np

def init():
    knn = cv2.ml.KNearest_create()
    img = cv2.imread(r'E:\python_code\firsttest\digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
    train = np.array(cells).reshape(-1,400).astype(np.float32)
    trainLabel = np.repeat(np.arange(10),500)
    return knn, train, trainLabel

def readinit():
    knn = cv2.ml.KNearest_create()
    with np.load('data.npz') as data:
        train = data['train']
        trainLabel = data['trainLabel']
    return knn, train, trainLabel

def updateKnn(knn, train, trainLabel, newData=None, newDataLabel=None):
    if type(newData) != type(None) and type(newDataLabel) != type(None):
    #if type(newData)!= type(None) and type(newDataLabel)!= type(None):
        print(train.shape, newData.shape)
        newData = newData.reshape(-1,400).astype(np.float32)
        train = np.vstack((train,newData))
        trainLabel = np.hstack((trainLabel,newDataLabel))
    knn.train(train,cv2.ml.ROW_SAMPLE,trainLabel)
    return knn, train, trainLabel



def shibie_num(knn, roi, thresValue):
    ret, th = cv2.threshold(roi, thresValue, 255, cv2.THRESH_BINARY)
    th = cv2.resize(th,(20,20))
    out = th.reshape(-1,400).astype(np.float32)
    # 根据knn算法，找到这个数字特征和训练样本的特征进行分类，识别出是哪个数字
    ret, result, neighbours, dist = knn.findNearest(out, k=5)
    return int(result[0][0]), th


knn, train, trainLabel = readinit()
#knn, train, trainLabel = init()
knn, train, trainLabel = updateKnn(knn, train, trainLabel)
#内置摄像头1，外置0
cap = cv2.VideoCapture(0)
width = 426*2
height = 480
videoFrame = cv2.VideoWriter('frame.avi',cv2.VideoWriter_fourcc('M','J','P','G'),25,(int(width),int(height)),True)
count = 0
while True:
    ret, frame = cap.read()
    frame = frame[:,:426]
    # 读取每一帧画面
    rois = []
    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 进行形态学膨胀和腐蚀，然后通过cv2.adsdiff(A, B)两幅图像作差，找到边
    gray2 = cv2.dilate(gray, None, iterations=2)
    gray2 = cv2.erode(gray2, None, iterations=2)
    edges = cv2.absdiff(gray, gray2)
    # 运用Sobel算子边缘检测
    x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 设置一个阈值来对图像进行分类
    ret, ddst = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
    # 找图片的轮廓
    im, contours, hierarchy = cv2.findContours(ddst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 20:
            rois.append((x, y, w, h))
    digits = []
    for r in rois:
        x, y, w, h = r
        digit, th = shibie_nim(knn, edges[y:y+h,x:x+w], 50)
        digits.append(cv2.resize(th,(20,20)))
        # 用矩形画出这个识别数字再写出这个识别数字
        cv2.rectangle(frame, (x,y), (x+w,y+h), (200,200,0), 2)
        cv2.putText(frame, str(digit), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2)
        print(str(digit))
    newEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    newFrame = np.hstack((frame,newEdges))
    cv2.imshow('frame', newFrame)
    videoFrame.write(newFrame)
    key = cv2.waitKey(1) & 0xff
    if key == ord(' '):
        break
    # 按s保存当前的数据
    elif key == ord('s'):
        np.savez('data.npz', train=train, trainLabel=trainLabel)
        print('保存数据成功')
    elif key == ord('u'):
        Nd = len(digits)
        n = len(digits)
        output = np.zeros(20 * 20 * n).reshape(-1, 20)
        for i in range(n):
            output[20 * i:20 * (i + 1), :] = digits[i]
        showDigits = cv2.resize(output,(60,60*Nd))
        cv2.imshow('digits', showDigits)
        cv2.imwrite(str(count)+'.png', showDigits)
        cv2.imwrite(r'C:\Users\win\Desktop\test\test3.JPG',frame)
        count += 1
        if cv2.waitKey(0) & 0xff == ord('e'):
            pass
        print('输入数字:')
        numbers = input().split(' ')
        Nn = len(numbers)
        if Nd != Nn:
            print('update fail!')
            continue
        try:
            for i in range(Nn):
                numbers[i] = int(numbers[i])
        except:
            continue
        knn, train, trainLabel = updateKnn(knn, train, trainLabel, output, numbers)
        print('update Done!')
cap.release()
cv2.destroyAllWindows()
