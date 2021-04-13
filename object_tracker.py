import cv2
#from YOLOv4 import detect

# 트랙커 객체 생성자 함수 리스트 ---①
trackers = [cv2.TrackerBoosting_create,
            cv2.TrackerMIL_create,
            cv2.TrackerKCF_create,
            cv2.TrackerTLD_create,
            cv2.TrackerMedianFlow_create,
            cv2.TrackerGOTURN_create, #버그로 오류 발생
            cv2.TrackerCSRT_create,
            cv2.TrackerMOSSE_create]
trackerIdx = 0  # 트랙커 생성자 함수 선택 인덱스
tracker = None
isFirst = True

bbox_log = []
x_speed_log = []
y_speed_log = []
x_accel_log = []
y_accel_log = []
frame_n = 0
speed_frame = 0
video_src = 0 # 비디오 파일과 카메라 선택 ---②
video_src = "./video/blackbox_trim.mp4"
cap = cv2.VideoCapture(video_src)
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)
win_name = 'Tracking APIs'
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Cannot read video file')
        break
    img_draw = frame.copy()
    if tracker is None: # 트랙커 생성 안된 경우
        cv2.putText(img_draw, "Press the Space to set ROI!!", \
            (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    else:
        
        ok, bbox = tracker.update(frame)   # 새로운 프레임에서 추적 위치 찾기 ---③
        (x,y,w,h) = bbox
        bbox_log.append(bbox)
        
        
        if ok: # 추적 성공
            cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), \
                          (0,255,0), 2, 1)
            cv2.putText(img_draw, "BBox Centerpoint : (%i,%i)"%(x,y), \
            (100,150), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
            frame_n = frame_n + 1 
 
            if frame_n > 4:
                x_speed = list(bbox_log[frame_n - 1])[0] - list(bbox_log[frame_n-4])[0]
                y_speed = list(bbox_log[frame_n - 1])[1] - list(bbox_log[frame_n-4])[1]
                x_speed_log.append(x_speed)
                y_speed_log.append(y_speed)
                speed_frame += 1
                cv2.putText(img_draw, "BBox speed : (%i,%i)"%(x_speed, y_speed), \
                (100,200), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)

            if speed_frame > 1:
                x_accel = (x_speed_log[speed_frame - 1] - x_speed_log[speed_frame-2]) * 10
                y_accel = (y_speed_log[speed_frame - 1] - y_speed_log[speed_frame-2]) * 10
                x_accel_log.append(x_accel)
                y_accel_log.append(y_accel)
                cv2.putText(img_draw, "BBox accel : (%i,%i)"%(x_accel, y_accel), \
                (100,250), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
                if (x_accel + y_accel < -100) or (x_accel + y_accel > 200):
                     cv2.putText(img_draw, "Road Accident Occur!", \
                (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
        
         
        else: # 추적 실패
            cv2.putText(img_draw, "Object Missing.", (100,80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    trackerName = tracker.__class__.__name__
    #cv2.putText(img_draw, str(trackerIdx) + ":"+trackerName , (500,20), \
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2,cv2.LINE_AA)

    cv2.imshow(win_name, img_draw)
    key = cv2.waitKey(delay) & 0xff
    # 스페이스 바 또는 비디오 파일 최초 실행 ---④
    if key == ord(' ') or (video_src != 0 and isFirst): 
        isFirst = False
        roi = cv2.selectROI(win_name, frame, False)  # 초기 객체 위치 설정
        if roi[2] and roi[3]:         # 위치 설정 값 있는 경우
            tracker = trackers[trackerIdx]()    #트랙커 객체 생성 ---⑤
            print(roi)
            isInit = tracker.init(frame, roi)
    elif key in range(48, 56): # 0~7 숫자 입력   ---⑥
        trackerIdx = key-48     # 선택한 숫자로 트랙커 인덱스 수정
        if bbox is not None:
            tracker = trackers[trackerIdx]() # 선택한 숫자의 트랙커 객체 생성 ---⑦
            isInit = tracker.init(frame, bbox) # 이전 추적 위치로 추적 위치 초기화
    elif key == 27 : 
        break
else:
    print( "Could not open video")

#print(x_accel_log)
cap.release()
cv2.destroyAllWindows()