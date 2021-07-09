from ctypes import *
"""
Ctypes를 이용한 Python과 C와의 연결

속도나 하드웨어 디바이스 드라이버 접근 등을 위해서 C와 연결해서 Python을 사용해야 할 일들은 생기기 마련이다.
두 언어를 이어 붙이기 위한 수단인 이종 언어간 인터페이스 (Foreign Function Interface, FFI)를 다룬다.

FFI를 지원하는 방법으로는 ctypes와 CFFI 방법이 있다.

ctypes방법은 파이썬 표준 라이브러리에 포함되어 있다.
C로 작성된 코드가 Shared object (*.so)형태로 컴파일 되어 있어야함.
함수와 구초제 등을 ctypes에 맞는 방식으로 다시 정리해야된다.
OS에서 제공하는 symbol lookup 방식으로 찾을 수 있는 요소에 한해서만 접근이 가능함

이를 통해 darknet 폴더내 C 코드를 동적 라이브러리 형태로 호출하여 사용가능
"""
import random
# box color 설정에 사용될 random.seed을 위한 library, 박스 플롯 색상을 랜덤으로 바꿔주는 역할
import os
# file path 확인
import cv2
# opencv-python
import time
# fps 계산을 위한 library

import darknet
""" 
darknet libdarknet.so(동적 라이브러리, make 파일 결과물) 호출 및 
detect, drawing에 필요한 함수가 들어있는 py 파일
 """

import argparse
# terminal 또는 prompt에서 사용할 명령어를 인식하기 위한 라이브러리

from threading import Thread, enumerate
"""
darknet webcam version는 1 process 3 Thread 형식으로 작동되는데
그때 사용할 Thread library 
"""

from queue import Queue
"""
Queue를 mansize를 1로 설정하여 생성하고

1 frame 호출 마다 Queue에 담아 순차별로 inference, drawing 과정을 진행
"""


def parser():
    """
    * parser()
        - yolov 구동에 필요한 파일들의 경로를 default로 둬서 바로 가져와서 실행할 수 있도록 만듦.
        - 인자값을 받을 수 있는 인스턴스(객체) 생성 argparse.ArgumentParser()
        - parameter: description = 인자 도움말 전에 표시할 텍스트
    """

    """ ====================================================================================

    * argparse
    : python script를 터미널에서 실행할 때 명령어 옵션에 대한 parameter를 python으로 전달할 때 사용하는 방
        사용예 : $./argv_test.py [param1][param2]
                parameter를 입력하지 않으면 default가 출력

    ===================================================================================="""

    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    """
         입력받을 인자값 등록 parser.add_argument()
         parameter:
         - name or flags : 옵션 문자역의 이름이나 리스트. --input
         - type : 명령행 인자가 변환되어야 할 형
         - default : 인자가 명령행에 없는 경우 생성되는 값
         - help : 인자가 하는 일에 대한 간단한 설명
     """
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov3.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov3.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    """
           명령창에서 주어진 인자를 파싱한다. parse_args()
           이 후 parser()로 인자값을 받아서 사용할 수 있다.
    """
    return parser.parse_args()


def str2int(video_path):
    """
    * str2int()
        - 받아오는 string type의 video_path를 int로 변환
        - ValueError(데이터 타입이 맞지 않을 때 발생하는 에러) 시 그대로 return
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    """
     * check_arguments_errors()
         - def parse()를 args로 받아와서 error를 검사
         - os.path.exists()로 받아오는 argument의 경로에 해당하는 파일이 존재하는를지 검사
         - 없다면 raise문으로 들어가서 예외처리로 들어감
     """

    """ ====================================================================================
    * assert 조건, '메세지' 
        : 가정 설정문 , assert 뒤에 조건에 만족하지 않으면 AssertionError 출력
    * raise 예외객체(예외내용)
        : 강제 예외처리 방법
    ===================================================================================="""

    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    """
            여기서 동영상 파일명,프레임,영상의 크기등을 지정하여
            video.write(image) 코드로 image 프레임을 저장.
    """
    """ ====================================================================================
    * cv2.VideoWriter_fourcc(*"코덱")
        : 디지털 미디어 포맷 코드를 생성, 인코딩 방식을 설정. MJPG 사용

    * cv2.CAP_PROP_FPS
        : 초당 비디오 프레임 호출 

    * cv2.VideoWriter(outfile, fourcc, frame, size)
        : 영상을 저장하기 위한 object
          저장될 파일명, Codec정보(cv2.VideoWriter_fourcc), 초당 저장될 frame, 저장될 사이즈(), 컬러 저장 여부
    ===================================================================================="""

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def video_capture(frame_queue, darknet_image_queue):
    
    while cap.isOpened(): # cap(VidedCapture object, 비디오 read 함수)가 비디오를 읽고있을때 loop 
        ret, frame = cap.read() # video에서 1 frame 씩 호출
        # ret(bool) - 비디오 호출 확인, 정상적 read - True, 호출하지 못했다면 False
        if not ret: # frame을 받아오지 못한다면 break
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # frame을 BGR에서 RGB형태로 변환
        frame_resized = cv2.resize(frame_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        """
         yolo network에 data를 input하기 위해
         어떤 video frame이 들어와도 yolo-cfg 파일에서 설정한  width, height로 resize
        """
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        """
        // Fast copy data from a contiguous byte array into the image.
        LIB_API void copy_image_from_bytes(image im, char *pdata)
        {
            unsigned char *data = (unsigned char*)pdata;
            int i, k, j;
            int w = im.w;
            int h = im.h;
            int c = im.c;
            for (k = 0; k < c; ++k) {
                for (j = 0; j < h; ++j) {
                    for (i = 0; i < w; ++i) {
                        int dst_index = i + w * j + w * h*k;
                        int src_index = k + c * i + c * w*j;
                        im.data[dst_index] = (float)data[src_index] / 255.;
                    }
                }
            }
        }
        빈 darknet_image 파일에 frame_resized을 byte화 해서 저장
        """
        darknet_image_queue.put(darknet_image)
        # inference에 사용할 Queue(maxsize=1)인 object에 byte화 한 frame을 저장
        frame_queue.put(frame_resized)
        # drawing에 사용할 Queue(maxsize=1)인 object에 frame을 저장

    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get() # videoCapture 함수에서 만든 byte화 한 frame 호출
        prev_time = time.time() # fps 계산을 위한 현재 time
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        """
        network - cfg 파일에 저장된 parameter 및 convolution network load
        class_name - names 파일내에 설정한 객체를 호출
        darknet_image = video_Capture 함수에서 만든 byte화 한 frame
        3가지 변수를 detect_image 함수에 넣으면
        darknet C 코드내에 network_predict_image 함수에서 input값인 image와
        cfg 파일내에 CNN을 통해 detection 결과값 도출(아래의 c언어 내부 코드를 사용하여)
        ---------------------------------------------------------------------------------------------------------------------
        detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
        {
            detection *dets = make_network_boxes(net, thresh, num);
            fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
            return dets;
        }
        
        detection *make_network_boxes(network *net, float thresh, int *num)
        {
            layer l = net->layers[net->n - 1];
            int i;
            int nboxes = num_detections(net, thresh);
            if(num) *num = nboxes;
            detection *dets = calloc(nboxes, sizeof(detection));
            for(i = 0; i < nboxes; ++i){
                dets[i].prob = calloc(l.classes, sizeof(float));
                if(l.coords > 4){
                    dets[i].mask = calloc(l.coords-4, sizeof(float));
                }
            }
            return dets;
        }
        
        void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
        {
            int j;
            for(j = 0; j < net->n; ++j){
                layer l = net->layers[j];
                if(l.type == YOLO){
                    int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
                    dets += count;
                }
                if(l.type == REGION){
                    get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
                    dets += l.w*l.h*l.n;
                }
                if(l.type == DETECTION){
                    get_detection_detections(l, w, h, thresh, dets);
                    dets += l.w*l.h*l.n;
                }
                
        }
        ----------------------------------------------------------------------------------------------------------------
        detection 값은 list형식으로 return 되고
        list내에 value는 tuple 형식으로 (label, 정확도, (박스 중점 x, 박스 중점 y, 박스 너비, 박스 높이)) 형식으로 출력
        """
        detections_queue.put(detections) # drawing에 사용할 결과값 
        fps = int(1/(time.time() - prev_time)) # 1/(현재 타임 - yolo detection전 타임) = fps
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        #Console detection 결과값 출력
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (width, height))
    #결과 Video 저장시 사용
    while cap.isOpened():
        frame_resized = frame_queue.get() # video_Capture 함수에서 원본 frame 호출
        detections = detections_queue.get() # inference 함수에서 detection 결과값 호출
        fps = fps_queue.get()
        if frame_resized is not None:
            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            """
            detection 결과값과 원본 프레임, 출력 label별 컬러 input
            def draw_boxes(detections, image, colors):
                for label, confidence, bbox in detections:
                    left, top, right, bottom = bbox2points(bbox)
                    cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
                    cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                                (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                colors[label], 1)
                return image
                
            해당함수에서 detection value값을 가지고 원본 프레임에 drawing   
            """
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # box가 drawing된 frame RGB화
            if args.out_filename is not None:
                video.write(image)
            if not args.dont_show:
                cv2.imshow('Inference', image)
                # 최종 화면상 출력부분, 원본 프레임에 박스플롯을 그려서 연속적으로 출력
            if cv2.waitKey(fps) == 27:
                break
    cap.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    #darknet 각 함수에 frame을 적용시킬때 사용하는 Queue

    args = parser() #terminal, prompt에서 구동시키기 위한 parser 활성화
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    """
    parser에서 작성한 cfg, data, weights file path를 가지고 
    load_network
    최초 실행시 cfg weight 활성화
    """
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    """
    cfg에서 설정한 width, height 값을 불러오는 함수
    동적라이브러리를 이용하여 해당 값을 호출
    """
    
    darknet_image = darknet.make_image(width, height, 3)
    """
    image.c내에 make_image 함수 호출
    image make_empty_image(int w, int h, int c)
    {
        image out;
        out.data = 0;
        out.h = h;
        out.w = w;
        out.c = c;
        return out;
    }
    
    image make_image(int w, int h, int c)
    {
        image out = make_empty_image(w,h,c);
        out.data = (float*)xcalloc(h * w * c, sizeof(float));
        return out;
    }
    
    빈 이미지 생성
    """
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(0)
    """
    동영상 - cv2.VideoCapture("동영상파일경로")
    webcam - cv2.VideoCapture(0)
    해당 opencv object로 video를 읽을 수 있음    
    """


    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
    """
    darknet python version은
    1 process 3 Thread 형식으로 동작
    Video_capture, inference, drawing 세개의 Thread를 통해
    캡쳐 -> 추론 -> 결과 drawing 세개의 단계를 거치며
    frame 추출 -> detection 결과 산출 -> detection 결과 그리기의 과정을 가짐
    해당 argument들은 main 작성 시 전역변수 Queue로 생성하여 해당 py 파일 전체에서 사용할수 있도록 작성     
    """

    """
    darknet algorithm flow
    main문을 시작으로
    1. parser load
    2. network load(parser data를 활용한 cfg, data,weights 파일 load)
    3. width,height load(cfg 정보를 활용)
    4. opencv VideoCapture load(video or webcam load)
    5. Thread 실행 Video_capture, inference, drawing
       Thread-(1) video_capture : 1 frame 씩 호출, queue append 
       Thread-(2) inference : queue에서 받은 frame으로 detection 결과값 도출
       Thread-(3) drawing : video_capture에서 가져온 frame과 inference의 detection 결과값으로
                     해당 frame에 box를 draw
        해당 (1), (2), (3)을 VideoCapture object가 close 될때까지 반복 
    
    각각 함수의 기능과 코드 세부사항은 해당 함수에 작성 
    """
