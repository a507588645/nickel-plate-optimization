import cv2
import numpy as np
import os
import math
import json
import copy

from app_paths import debug_file_path
from app_settings import get_app_settings


APP_SETTINGS = get_app_settings()


def _write_debug_file(filename, content, mode='w'):
    if not APP_SETTINGS.runtime.debug_artifacts_enabled:
        return
    with open(debug_file_path(filename), mode, encoding='utf-8') as f:
        f.write(content)

# 读取图片
def readimg(src):
    img = cv2.imread(src)
    #if 'bottom' in src:
        #img = cv2.flip(img,1)
    image = cv2.resize(img,(1024,888))
    return image

def readimgtb(src):
    img = cv2.imread(src)
    #if 'bottom' in src:
    img = cv2.flip(img,1)
    image = cv2.resize(img,(1024,888))
    return image

# 计算角度，宽高，中心算法
def findcent(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #90，125关系到镍板轮廓查找
    _, binary = cv2.threshold(gray,40,130,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for c in range(len(contours)):
        areas.append(cv2.contourArea(contours[c]))
    max_id = areas.index(max(areas))
    max_rect = cv2.minAreaRect(contours[max_id])
    box = cv2.boxPoints(max_rect)
    # 这一步不影响后面的画图，但是可以保证四个角点坐标为顺时针
    startidx = box.sum(axis=1).argmin()
    box = np.roll(box,4-startidx,0)
    # 在原图上计算预测的外接矩形
    box = box.reshape((-1,1,2)).astype(np.int32)

    #外接矩形添加偏移量
    box[2][0][1] -= 55
    box[3][0][1] -= 55

    rotheta = math.atan((box[1][0][1] - box[0][0][1])/(box[1][0][0] - box[0][0][0]))
    theta = int(rotheta*180/math.pi)
    
    width = int((box[1][0][0] - box[0][0][0])/math.cos(abs(theta)))
    height = int((box[3][0][1] - box[0][0][1])/math.cos(abs(theta)))
    centerx = (box[0][0][0] + box[1][0][0] + box[2][0][0] + box[3][0][0])/4
    centerx = int(centerx)
    centery = (box[0][0][1] + box[1][0][1] + box[2][0][1] + box[3][0][1])/4
    centery = int(centery) 
    
    # center = (int(centerx),int(centery))
    center = (centerx,centery)
    return theta,width,height,center


# 旋转及输出中心坐标角度算法
def xuanZhuan(input_imagepath,out_imagepath):
    ogwidth = 4096
    ogheight = 3554
    towidth = 1024
    toheight = 888
    kernel = np.ones((7,7), np.uint8)
    path = input_imagepath
    img = readimg(path)
    msg = findcent(img)
    #print('msg'+ str(msg))
    savePath = os.path.join(out_imagepath, os.path.basename(path))
    M=[]
    theta = msg[0]
    center = msg[3]
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    x, y = img.shape[0:2]
    M = cv2.getRotationMatrix2D((y/2,x/2), theta, 1)
    rotated = cv2.warpAffine(img, M, (y, x))
    rotated = cv2.resize(rotated,(towidth,toheight))
    aftermsg = findcent(rotated)
    dx = towidth/2 - aftermsg[3][0]
    dy = toheight/2 - aftermsg[3][1] 
    MAT = np.float32([[1, 0, dx], [0, 1, dy]])  # 构造平移变换矩阵   
    moveimg = cv2.warpAffine(rotated, MAT, (towidth, toheight))  # 默认为黑色填充
    scaleimg = cv2.resize(moveimg,(ogwidth,ogheight))
    cv2.imencode('.png',scaleimg)[1].tofile(savePath)
    #图像尺寸1024*888 若镍板宽度小于500 或宽度大于950 或高度小于500
    # if aftermsg[1] < 500 or aftermsg[1]>950 or aftermsg[2] <500:
    #     warnmsg = '图像异常'
    #     return warnmsg
    # else:
    aftermsg = list(aftermsg)
    msg = list(msg)
    aftermsg[0] = msg[0]
    return aftermsg

def xuanZhuan_and_flip(input_imagepath,out_imagepath):
    ogwidth = 4096
    ogheight = 3554
    towidth = 1024
    toheight = 888
    kernel = np.ones((7,7), np.uint8)
    path = input_imagepath
    img = readimg(path)
    msg = findcent(img)
    #print('msg'+ str(msg))
    savePath = os.path.join(out_imagepath, os.path.basename(path))
    M=[]
    theta = msg[0]
    center = msg[3]
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    x, y = img.shape[0:2]
    M = cv2.getRotationMatrix2D((y/2,x/2), theta, 1)
    rotated = cv2.warpAffine(img, M, (y, x))
    rotated = cv2.resize(rotated,(towidth,toheight))
    aftermsg = findcent(rotated)
    dx = towidth/2 - aftermsg[3][0]
    dy = toheight/2 - aftermsg[3][1] 
    MAT = np.float32([[1, 0, dx], [0, 1, dy]])  # 构造平移变换矩阵   
    moveimg = cv2.warpAffine(rotated, MAT, (towidth, toheight))  # 默认为黑色填充
    moveimg = cv2.flip(moveimg,1)
    scaleimg = cv2.resize(moveimg,(ogwidth,ogheight))
    cv2.imencode('.png',scaleimg)[1].tofile(savePath)
    #图像尺寸1024*888 若镍板宽度小于500 或宽度大于950 或高度小于500
    # if aftermsg[1] < 500 or aftermsg[1]>950 or aftermsg[2] <500:
    #     warnmsg = '图像异常'
    #     return warnmsg
    # else:
    aftermsg = list(aftermsg)
    msg = list(msg)
    aftermsg[0] = msg[0]
    return aftermsg

#  缺陷面积求和函数，array为矩形的坐标矩阵，(1024,888为图片尺寸)
def scalculate(array):
    a = np.zeros((4069, 3554))
    for i in array:
        x1 = int(i[0])
        y1 = int(i[1])
        x2 = int(i[2])
        y2 = int(i[3])
        for j in range(x1, x2):
            for k in range(y1, y2):
                a[j][k] = 1
    sums = int(np.sum(a))
    return sums

#求切一刀时相交瑕疵坐标列表
def bads(arrs,i,p):      #i=0,1,2,3表示靠左，右，上，下侧切
    arrscopy = copy.deepcopy(arrs)
    barrs = []
    if i == 0:
        for arr in arrscopy:
            if arr[0] >= p:
                continue
            if arr[2] >= p:
                arr[2] = p
            barrs.append(arr)
    elif i == 1:
        for arr in arrscopy:
            if arr[2] <= p:
                continue
            if arr[0] <= p:
                arr[0] = p
            barrs.append(arr)
    elif i == 2:
        for arr in arrscopy:
            if arr[1] >= p:
                continue
            if arr[3] >= p:
                arr[3] = p
            barrs.append(arr)
    elif i == 3:
        for arr in arrscopy:
            if arr[3] <= p:
                continue
            if arr[1] <= p:
                arr[1] = p
            barrs.append(arr)
    return barrs

#求横切五刀时各条相交瑕疵坐标列表
def nietiaobads(arrs,p1,p2):      
    arrscopy = copy.deepcopy(arrs)
    badarrs = []
    for arr in arrscopy:
        if arr[1] >= p2 or arr[3] <= p1:
            continue
        if arr[1] <= p1:
            arr[1] = p1
        if arr[3] >= p2:
            arr[3] = p2
        badarrs.append(arr)
    return badarrs

#求竖切时各镍条相交瑕疵坐标列表
def nietiaobadsy(arrs,p1,p2):      
    arrscopy = copy.deepcopy(arrs)
    badarrs = []
    for arr in arrscopy:
        if arr[0] >= p2 or arr[2] <= p1:
            continue
        if arr[0] <= p1:
            arr[0] = p1
        if arr[2] >= p2:
            arr[2] = p2
        badarrs.append(arr)
    return badarrs

#读取json文件 获得除水渍外各类型缺陷左上，右下顶点坐标list,转为当前图坐标系
#noduletype 0合格无结瘤 1合格有结瘤 2烫洗 3其他
def readarray(json_path):
    plist = []
    nodule_info = []
    json_path = json_path.replace("'", "\"")
    json_path=json.loads(json_path)
    #镍板图像宽高
    banw = 4096
    banh = 3554
    #图片尺寸
    tow = 1024
    toh = 888
    ndtype = json_path["nodule"]
    
    for i in json_path["defects"]:
        # 边部结瘤
        if i['label'] == 3:
            badornot = True 
            px1 = i['cx'] - i['w']/2
            px1 = int(px1/4)
            py1 = int(i['cy'] - i['h']/2)
            py1 = int(py1/4)
            px2 = int(i['cx'] + i['w']/2)
            px2 = int(px2/4)
            py2 = int(i['cy'] + i['h']/2)
            py2 = int(py2/4)
            nodule_info.append([px1,py1,px2,py2])
            # #下
            # if py1 >= 444 and py2 >= 444:
            #     noduleedge = 1
            # #左
            # if px1 <= 512 and px2 <= 512:
            #     noduleedge = 4
            # #上
            # if py1 <= 444 and py2 <=444:
            #     noduleedge = 3
            # #右
            # if px1 >= 512 and px2 >= 512:
            #     noduleedge = 2
            # print(px1,py1,px2,py2)

        # 非污渍、水渍 
        if i['label']!=0 and i['label']!=1:
            px1 = i['cx'] - i['w']/2
            px1 = int(px1/4)
            py1 = int(i['cy'] - i['h']/2)
            py1 = int(py1/4)
            px2 = int(i['cx'] + i['w']/2)
            px2 = int(px2/4)
            py2 = int(i['cy'] + i['h']/2)
            py2 = int(py2/4)
            p = [px1,py1,px2,py2]
            plist.append(p)
    _write_debug_file('plistdebug.txt', 'plist='+str(plist)+'\r\n', mode='a')
    return plist,nodule_info,ndtype

# 发乌、烧板、酸污染list
def readarray_C(json_path):
    plist = []
    json_path = json_path.replace("'", "\"")
    json_path=json.loads(json_path)
    for i in json_path["defects"]:
        if i['label'] == 7 or i['label'] == 8 or i['label'] == 11:
            px1 = i['cx'] - i['w']/2
            px1 = int(px1/4)
            py1 = int(i['cy'] - i['h']/2)
            py1 = int(py1/4)
            px2 = int(i['cx'] + i['w']/2)
            px2 = int(px2/4)
            py2 = int(i['cy'] + i['h']/2)
            py2 = int(py2/4)
            p = [px1,py1,px2,py2]
            plist.append(p)
    return plist

def read_watermark(json_path):
    plist = []
    json_path = json_path.replace("'", "\"")
    json_path=json.loads(json_path)
    for i in json_path["defects"]:
        if i['label'] == 13:
            px1 = i['cx'] - i['w']/2
            px1 = int(px1/4)
            py1 = int(i['cy'] - i['h']/2)
            py1 = int(py1/4)
            px2 = int(i['cx'] + i['w']/2)
            px2 = int(px2/4)
            py2 = int(i['cy'] + i['h']/2)
            py2 = int(py2/4)
            p = [px1,py1,px2,py2]
            plist.append(p)
    return plist

# 缺角的list
def read_nocorner(json_path):
    plist = []
    json_path = json_path.replace("'", "\"")
    json_path=json.loads(json_path)
    for i in json_path["defects"]:
        if i['label'] == 12:
            px1 = i['cx'] - i['w']/2
            px1 = int(px1/4)
            py1 = int(i['cy'] - i['h']/2)
            py1 = int(py1/4)
            px2 = int(i['cx'] + i['w']/2)
            px2 = int(px2/4)
            py2 = int(i['cy'] + i['h']/2)
            py2 = int(py2/4)
            p = [px1,py1,px2,py2]
            plist.append(p)
    return plist

#读取yolov5输出output 获得污渍、水渍缺陷左上，右下顶点坐标list,转为当前图坐标系
def read_A(output):
    plist = []
    for i in output:
        if i['label']==0 or i['label']==1:
            px1 = int(i['cx']-i['w']/2)
            px1 = int(px1/4)
            py1 = int(i['cy'] - i['h']/2)
            py1 = int(py1/4)
            px2 = int(i['cx'] + i['w']/2)
            px2 = int(px2/4)
            py2 = int(i['cy'] + i['h']/2)
            py2 = int(py2/4)
            p = [px1,py1,px2,py2]
            plist.append(p)
    return plist

# 黄渍、气孔、结粒、疙瘩、绿色结晶、缺角、严重水渍
def read_B(output):
    plist = []
    for i in output:
        if i['label']==2 or i['label']==4 or i['label']==5 or i['label']==6 or i['label']==10 or i['label']==12 or i['label']==13:
            px1 = int(i['cx']-i['w']/2)
            px1 = int(px1/4)
            py1 = int(i['cy'] - i['h']/2)
            py1 = int(py1/4)
            px2 = int(i['cx'] + i['w']/2)
            px2 = int(px2/4)
            py2 = int(i['cy'] + i['h']/2)
            py2 = int(py2/4)
            p = [px1,py1,px2,py2]
            plist.append(p)
    return plist

# 发乌、烧板、酸污染
def read_C(output):
    plist = []
    for i in output:
        if i['label']==7 or i['label']==8 or i['label']==11:
            px1 = int(i['cx']-i['w']/2)
            px1 = int(px1/4)
            py1 = int(i['cy'] - i['h']/2)
            py1 = int(py1/4)
            px2 = int(i['cx'] + i['w']/2)
            px2 = int(px2/4)
            py2 = int(i['cy'] + i['h']/2)
            py2 = int(py2/4)
            p = [px1,py1,px2,py2]
            plist.append(p)
    return plist


def read_airhole(output):
    plist = []
    for i in output:
        if i['label']==4:
            px1 = int(i['cx']-i['w']/2)
            px1 = int(px1/4)
            py1 = int(i['cy'] - i['h']/2)
            py1 = int(py1/4)
            px2 = int(i['cx'] + i['w']/2)
            px2 = int(px2/4)
            py2 = int(i['cy'] + i['h']/2)
            py2 = int(py2/4)
            p = [px1,py1,px2,py2]
            plist.append(p)
    return plist

#上下表面缺陷合并算法，平移及放缩
def bioplist(center1,center2,size1,size2,plist1,plist2):
    tow = 1024
    toh = 888
    # xshift = center1[0] - center2[0]
    # yshift = center1[1] - center2[1]
    xshift = 0
    yshift = 0
    scale = 1
    for p in plist2:
        p[0] = min(int((p[0] + xshift)*scale),tow - 1)
        p[0] = max(p[0],0)
        p[1] = min(int((p[1] + yshift)*scale),tow - 1)
        p[1] = max(p[1],0)
        p[2] = min(int((p[2] + xshift)*scale),toh - 1)
        p[2] = max(p[2],0)
        p[3] = min(int((p[3] + yshift)*scale),toh - 1)
        p[3] = max(p[3],0)
    plist1.extend(plist2)
    _write_debug_file('debug.txt', 'plist1='+str(plist1)+'\r\n'+'plist2'+str(plist2))
    return plist1

#划线算法
#type=0，无板;1，3#; 2，烫洗板; 3，合格整版; 4，合格600; 5，合格不足600; 6, 2#
def cutmsg(center,array,array_C,noduleedges,nocorner,ndtype):
    #镍板图像长宽
    banw = 4096
    banh = 3554
    #缩放图像宽高
    tow = 1024
    toh = 888
    #计算缺陷面积
    areas = scalculate(array) #4096*3554
    areas_C = scalculate(array_C) #4096*3554
    areas_nocorner = scalculate(nocorner) #4096*3554
    snieban = 720*740
    # wslice = width/6
    wslice = 120
    #10为偏移量 实际中可能需修改
    # hslice = height/6 - 10
    hslice = 125
    #分情况处理
    #1垂直 2水平 1合格 2坏的
    clist =  {'type':0,'cutting_count': 0, 'direction': 0, 'positions': [],'states':[],'cut_edges':[]}
    # if ndtype == 0:
    #     clist['type'] = 3
    #     clist['cutting_count'] = 0
    _write_debug_file('s%.txt', 's%='+str(areas/snieban)+'\r\n'+'s%='+str(areas_C/snieban))

    if len(noduleedges) != 0 or areas_C >= snieban * 0.5 or areas_nocorner >0:
        clist['type'] = 1
        clist['direction'] = 0
        clist['cutting_count'] = 0
        clist['cut_edges'] = noduleedges

    elif areas/snieban >= 0.6:
    # if areas >= snieban * 0.6:
        clist['type'] = 6
        clist['direction'] = 0
        clist['cutting_count'] = 0
        clist['cut_edges'] = noduleedges
        
    elif areas <snieban * 0.6 and areas >= snieban * 1/10:
        clist['type'] = 5
        clist['direction'] = 1
        clist['cutting_count'] = 5
        # silce0 = center[0] - wslice*2
        # silce1 = center[0] - wslice
        # silce2 = center[0]
        # silce3 = center[0] + wslice
        # silce4 = center[0] + wslice*2
        silce0 = int((center[1] - hslice*2)*banh/toh)
        silce1 = int((center[1] - hslice)*banh/toh)
        silce2 = int((center[1])*banh/toh)
        silce3 = int((center[1] + hslice*1)*banh/toh)
        silce4 = int((center[1] + hslice*2)*banh/toh)
        slices = [silce0,silce1,silce2,silce3,silce4]
        slices = [ i + 20 for i in slices ]
        # slices = [ int(i*banh/toh) for i in slices]   #修改
        tiaobads0 = nietiaobads(array,0,slices[0]/4)
        tiaobads1 = nietiaobads(array,slices[0]/4,slices[1]/4)  
        tiaobads2 = nietiaobads(array,slices[1]/4,slices[2]/4)  
        tiaobads3 = nietiaobads(array,slices[2]/4,slices[3]/4)  
        tiaobads4 = nietiaobads(array,slices[3]/4,slices[4]/4)  
        tiaobads5 = nietiaobads(array,slices[4]/4,1000)  
        # tiaobads0 = nietiaobads(array,0,444 - 120*2)
        # tiaobads1 = nietiaobads(array,444 - 120*2,444 -120)  
        # tiaobads2 = nietiaobads(array,444 - 120,444)  
        # tiaobads3 = nietiaobads(array,444 ,444 + 120) 
        # tiaobads4 = nietiaobads(array,444 + 120,444 +240)  
        # tiaobads5 = nietiaobads(array,444 +240 ,800)  
        tiaos0 = scalculate(tiaobads0)
        tiaos1 = scalculate(tiaobads1)
        tiaos2 = scalculate(tiaobads2)
        tiaos3 = scalculate(tiaobads3)
        tiaos4 = scalculate(tiaobads4)
        tiaos5 = scalculate(tiaobads5)
        slicestxt = [i/4 for i in slices]
        _write_debug_file(
            'tiaodebug.txt',
            's%='+str(areas/snieban)+'\r\n'
            +'slicestxt='+str(slicestxt)+'\r\n'
            +'array='+str(array)+'\r\n'
            +'tiaobads0='+str(tiaobads0)+'\r\n'
            +'tiaobad1='+str(tiaobads1)+'\r\n'
            +'tiaobad2='+str(tiaobads2)+'\r\n'
            +'tiaobad3='+str(tiaobads3)+'\r\n'
            +'tiaobad4='+str(tiaobads4)+'\r\n'
            +'tiaobad5='+str(tiaobads5)+'\r\n'
            +'tiao0='+str(tiaos0)+'\r\n'
            +'tiao1='+str(tiaos1)+'\r\n'
            +'tiao2='+str(tiaos2)+'\r\n'
            +'tiao3='+str(tiaos3)+'\r\n'
            +'tiao4='+str(tiaos4)+'\r\n'
            +'tiao5='+str(tiaos5)+'\r\n'
        )
        gdorbd = [1,1,1,1,1,1]  
        if tiaos0 > 5000:
            gdorbd[0] = 2
        if tiaos1 > 5000:
            gdorbd[1] = 2
        if tiaos2 > 5000:
            gdorbd[2] = 2
        if tiaos3 > 5000:
            gdorbd[3] = 2
        if tiaos4 > 5000:
            gdorbd[4] = 2
        if tiaos5 > 5000:
            gdorbd[5] = 2
        clist['positions'] = slices
        clist['states'] = gdorbd
        # clist['states'].reverse()
        if np.sum(gdorbd) <7:
            clist['type'] = 3
            clist['direction'] = 0
            clist['cutting_count'] = 0
            clist['positions'] = []       
            clist['states'] = []
        elif np.sum(gdorbd[0:4]) <5:
            clist['type'] = 4
            clist['direction'] = 1
            clist['cutting_count'] = 2
            clist['positions'] = [int((center[1] + hslice)*banw/tow),int((center[1] + hslice*2)*banw/tow)]
            clist['states'] = [1,gdorbd[4],gdorbd[5]]
        elif np.sum(gdorbd[2:6]) <5:
            clist['type'] = 4
            clist['direction'] = 3
            clist['cutting_count'] = 2
            clist['positions'] = [int((center[1] - hslice*2)*banw/tow),int((center[1] - hslice)*banw/tow)]       
            clist['states'] = [gdorbd[0],gdorbd[1],1]
            clist['states'].reverse()
        clist['states'].reverse()
        if np.sum(gdorbd) > 9:
            clist['type'] = 6     #切不出合格的3条，降为2号板
            clist['direction'] = 0
            clist['cutting_count'] = 0
            clist['positions'] = []
            clist['states'] = []
        # 切五刀 列表
    elif areas < snieban * 1/10 and areas >snieban * 0.01:
        #切一刀  innerbads0、1、2、3表示靠左、右、上、下侧切
        clist['type'] = 4
        clist['cutting_count'] = 1

        innerbads3 = bads(array, 3, center[1] + hslice)   #20为偏移量 镍耳长度
        # innerbads3 = bads(array,3, 200)   #20为偏移量 镍耳长度
        s3 = scalculate(innerbads3)

        # innerbads0 = bads(array,0,0)
        innerbads0 = bads(array, 0, center[0] - wslice)
        s0 = scalculate(innerbads0)

        # innerbads1 = bads(array,1,700)
        innerbads1 = bads(array, 1, center[0] + wslice)
        s1 = scalculate(innerbads1)

        innerbads2 = bads(array, 2, center[1] - hslice)
        s2 = scalculate(innerbads2)      

        badlists = [s0,s1,s2,s3]
        nswe = badlists.index(max(badlists))
        #direction = 1/2/3/4  下/右/上/左  0/90/180/270
        #靠左切  
        # nswe = 0, 1，2，3为左右上下
        if nswe == 0:
            clist['direction'] = 4
            clist['cutting_count'] = 2
            clist['positions'] = [int((center[0] - wslice*2)*banw/tow),int((center[0] - wslice)*banw/tow)]
            clist['states'] = [2,2,1]
            cutpos = [i/4 for i in clist['positions']]
            tiaobads0 = nietiaobadsy(array,0,cutpos[0]) 
            tiaos0 = scalculate(tiaobads0)
            tiaobads1 = nietiaobadsy(array,cutpos[0],cutpos[1])  
            tiaos1 = scalculate(tiaobads1)
            if tiaos0 < 4000:
                clist['states'][0] =1
            if tiaos1 < 4000:
                clist['states'][1] =1
            # clist['states'].reverse()
            if np.sum(clist['states']) < 4:
                clist['type'] = 3
                clist['direction'] = 0
                clist['cutting_count'] = 0
                clist['positions'] = []       
                clist['states'] = []
            _write_debug_file(
                'leftbug.txt',
                'defects='+str(array)+'\r\n'
                +'cutpos='+str(cutpos)+'\r\n'
                +'tiaobads0='+str(tiaobads0)+'\r\n'
                +'tiaobads1='+str(tiaobads1)+'\r\n'
                +'tiaos0='+str(tiaos0)+'\r\n'
                +'tiaos1='+str(tiaos1)+'\r\n'
            )
        #靠右切
        elif nswe == 1:
            clist['direction'] = 2
            clist['cutting_count'] = 2
            clist['positions'] = [int((center[0] + wslice)*banw/tow),int((center[0] + wslice*2)*banw/tow)]
            clist['states'] = [1,2,2]
            cutpos = [i/4 for i in clist['positions']]
            tiaobads0 = nietiaobadsy(array,cutpos[0],cutpos[1]) 
            tiaos0 = scalculate(tiaobads0)
            tiaobads1 = nietiaobadsy(array,cutpos[1],1000)  
            tiaos1 = scalculate(tiaobads1)
            if tiaos0 < 4000:
                clist['states'][1] =1
            if tiaos1 < 4000:
                clist['states'][2] =1
            clist['states'].reverse()
            if np.sum(clist['states']) < 4:
                clist['type'] = 3
                clist['direction'] = 0
                clist['cutting_count'] = 0
                clist['positions'] = []       
                clist['states'] = []
            _write_debug_file(
                'rightbug.txt',
                'defects='+str(array)+'\r\n'
                +'cutpos='+str(cutpos)+'\r\n'
                +'tiaobads0='+str(tiaobads0)+'\r\n'
                +'tiaobads1='+str(tiaobads1)+'\r\n'
                +'tiaos0='+str(tiaos0)+'\r\n'
                +'tiaos1='+str(tiaos1)+'\r\n'
            )
        #靠上切
        elif nswe == 2:
            clist['direction'] = 3
            clist['cutting_count'] = 2
            clist['positions'] = [int((center[1] - hslice*2)*banh/toh),int((center[1] - hslice)*banh/toh)]
            clist['states'] = [2,2,1]
            cutpos = [i/4 for i in clist['positions']]
            tiaobads0 = nietiaobads(array,0,cutpos[0]) 
            tiaos0 = scalculate(tiaobads0)
            tiaobads1 = nietiaobads(array,cutpos[0],cutpos[1])  
            tiaos1 = scalculate(tiaobads1)
            if tiaos0 < 4000:
                clist['states'][0] =1
            if tiaos1 < 4000:
                clist['states'][1] =1
            # clist['states'].reverse()    
            if np.sum(clist['states']) < 4:
                clist['type'] = 3
                clist['direction'] = 0
                clist['cutting_count'] = 0
                clist['positions'] = []       
                clist['states'] = [] 
            _write_debug_file(
                'bottomcutbug.txt',
                'defects='+str(array)+'\r\n'
                +'cutpos='+str(cutpos)+'\r\n'
                +'tiaobads0='+str(tiaobads0)+'\r\n'
                +'tiaobads1='+str(tiaobads1)+'\r\n'
                +'tiaos0='+str(tiaos0)+'\r\n'
                +'tiaos1='+str(tiaos1)+'\r\n'
            )
        #靠下切
        elif nswe == 3:
            clist['direction'] = 1
            clist['cutting_count'] = 2
            clist['positions'] = [int((center[1] + hslice)*banh/toh),int((center[1] + hslice*2)*banh/toh)]
            clist['states'] = [1,2,2]
            cutpos = [i/4 for i in clist['positions']]
            tiaobads0 = nietiaobads(array,cutpos[0],cutpos[1]) 
            tiaos0 = scalculate(tiaobads0)
            tiaobads1 = nietiaobads(array,cutpos[1],900)  
            tiaos1 = scalculate(tiaobads1)
            if tiaos0 < 4000:
                clist['states'][1] =1
            if tiaos1 < 4000:
                clist['states'][2] =1
            clist['states'].reverse() 
            if np.sum(clist['states']) < 4:
                clist['type'] = 3
                clist['direction'] = 0
                clist['cutting_count'] = 0
                clist['positions'] = []       
                clist['states'] = []
        # if np.sum(clist['states']) <4:
        #     clist['type'] = 3
        #     clist['direction'] = 0
        #     clist['cutting_count'] = 0
        #     clist['positions'] = []       
        #     clist['states'] = []
    else:
        if ndtype == 3:
            clist['type'] = 2
        else:
            clist['type'] = 3
        clist['cutting_count'] = 0
    _write_debug_file('clistdebug.txt', 'array='+str(clist))
    return clist

#根据nodule缺陷区域左上右下坐标，和镍板区域左上右下坐标返回剪边信息
def jianbian(nodule_info, img_edge):
    nodule_number = len(nodule_info)
    nodule_edge=[]
    if nodule_number == 0:
        return nodule_edge
    
    else:

        image_px1 = img_edge[0]
        image_py1 = img_edge[1]
        image_px2 = img_edge[2]
        image_py2 = img_edge[3]

        for i in range(nodule_number):

            nodule_px1 = nodule_info[i][0]
            nodule_py1 = nodule_info[i][1]
            nodule_px2 = nodule_info[i][2]
            nodule_py2 = nodule_info[i][3]

            #下
            if abs(image_py2-nodule_py2) < 30 and abs(image_py2-nodule_py1) < 30:  
                nodule_edge.append(1)

            #右
            if abs(image_px2-nodule_px2) < 30 and abs(image_px2-nodule_px1) < 30:  
                nodule_edge.append(2)

            #上
            if abs(image_py1-nodule_py2) < 30 and abs(image_py1-nodule_py1) < 30:  
                nodule_edge.append(3)

            #左
            if abs(image_px1-nodule_px2) < 30 and abs(image_px1-nodule_px1) < 30:  
                nodule_edge.append(4)
    
    return nodule_edge


# def find_tangxiban()
#         defect_area=read_waterstain_wuzi_list(output)
#         defect_area=scalculate(defect_area)
#         result=findcent(image1)
#         width=result[1]
#         height=result[2]
#         nieban_area=width*height
#         waterstain_wuzi_ratio=defect_area/nieban_area
        #当水印，污渍合计面积占比大于0.5时，判断为烫洗板

if __name__ == '__main__':
    image="C:\\Users\\A\\Desktop\\nieban_pic\\20230105172454356_0_2.png"
    out_imagedirpath="C:\\Users\\A\\Desktop"

    xuanZhuan(image,out_imagedirpath)
    #cv2.waitKey(0)
