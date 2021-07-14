# Object_Tracker

OpenCV Based Object Tracking Code Baseline by BaekKyunShin, https://github.com/BaekKyunShin/OpenCV_Project_Python/tree/master/08.match_track

__Purpose : To detect car accident event with object tracking(classical)__

## 전체 목적 : 스마트 도로조명 활용 도시재난안전관리 연계 기술 개발 중 교통사고 검출  
이 중 Edge(CCTV, 스마트 도로조명등) 부분에서 검출할때, 현 코드와 같은 Object Tracking API(OpenCV 제공) or SORT와 같은 방식 고려중 (SORT Github : https://github.com/abewley/sort)  
직접적인 Server에서는 좀 더 무거운 FairMOT와 같은 방식으로 Multiple Object Tracking을 하여 검출하는 알고리즘 고려 (FairMOT Github : https://github.com/ifzhang/FairMOT)

현재는 아주 간단한 Task로 Frame 별 이동 거리를 바탕으로 (FPS는 일정할 것이므로) 속도, Frame 별 이동거리를 바탕으로 가속도를 구하여  
너무 급격한 변화(급정거, 급제동, Object 왜곡 등)를 탐지하여 Car Accident를 비추는 간단한 코드

Blackbox base의 영상으로 start, 시작된 영상에서 차량을 Drag 하여 Bounding Box 표시, 이를 바탕으로 사고, 비사고를 검출

ex) 좌측면에 있는 차량에 해당하게 Bounding Box를 손으로 하는 방식, 동작하는것은 다음과 같음

[![Demo](http://img.youtube.com/vi/tvtUrwRFx8E/0.jpg)](https://youtu.be/tvtUrwRFx8E) 

Further work :  
Object Detector + Object Tracker Based 교통사고 검출  


[![Demo](http://img.youtube.com/vi/ZmSdxTWi2es/0.jpg)](https://youtu.be/ZmSdxTWi2es)  
[![Demo](http://img.youtube.com/vi/SD5RrcX_89o/0.jpg)](https://youtu.be/SD5RrcX_89o)  
To be done
MOT in FairMOT, JDE ... (More Deep - Network Version)  
Accident detection network reinforcement :  
(1) Hard-coded,  
(2)deep learning based
