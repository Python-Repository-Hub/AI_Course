import glob
# folder 내에 file list 추출 library

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

import cv2
# opencv-python

import numpy as np

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
    parser.add_argument("--input", type=str, default="C:\Users\USER\Desktop\edu\vision3_od\mask_exam.mp4",
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="C:\Users\USER\Desktop\edu\vision3_od\yolov4-train09090_best.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="C:\Users\USER\Desktop\edu\vision3_od\yolov4-train0909.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="C:\Users\USER\Desktop\edu\vision3_od\golden2.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    """
           명령창에서 주어진 인자를 파싱한다. parse_args()
           이 후 parser()로 인자값을 받아서 사용할 수 있다.
    """
    return parser.parse_args()


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
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    # width, height load(cfg 정보를 활용)

    darknet_image = darknet.make_image(width, height, 3)
    # yolo에 사용될 width, height 크기에 맞는 빈 이미지 생성

    image = cv2.imread(image_path)
    #  image_path에 존재하는 image file load

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image를 BGR → RGB 변환
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    # yolo에 사용될 width, height 크기에 맞게 image resize

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    # 빈 이미지와 resize image를 사용하여 빈 이미지를 yolo에 사용될 byte 형태로 이미지를 값을 채움
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    """ 
    cfg, class_names, byte형태의 이미지를 토대로 yolo detection 실행하여 detection 값 도출
      ※ detection 값은 list형식으로 return 되고
        list내에 value는 tuple 형식으로 (label, 정확도, (박스 중점 x, 박스 중점 y, 박스 너비, 박스 높이)) 형식으로 출력 
    """

    darknet.free_image(darknet_image)
    # darknet image 메모리 할당 해제(C언어는 memory 할당 및 해제를 직접 해야함)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    """
       detection 값 내에 박스중점(x,y), 박스크기(width,height)를 사용하여 
       원본 이미지 내에 각각 detection box를 draw
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
    # main 함수로 검출된 객체의 box가 그려진 frame과 detection 결과 값을 반환

def main():
    args = parser()  # terminal, prompt에서 구동시키기 위한 parser 활성화
    check_arguments_errors(args)

    random.seed(3)  # box color 랜덤 결정 seed

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

    show_state = 0
    label_state = 1
    
    input_path = "./input" # data가 포함된 폴더
    images = load_images(input_path) # glob를 사용하여 파일 내 jpg, jpeg, png 파일 리스트 추출

    index = 0
    while True:
        if index >= len(images):
            break
        image_name = images[index]
        prev_time = time.time()
        image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
        )

        if not label_state:
            save_annotations(image_name, image, detections, class_names)
            
        """
        detection 결과를 labeling에 사용할 txt로 뽑아내는 함수
        label_state가 0일 시 txt 파일로 생성
        해당 함수를 통해 단순 detection 결과 값들을 Yolo 학습에 사용할 수 있게 라벨링 가능
        """

        darknet.print_detections(detections, args.ext_output)
        # detection 결과값 Console 출력

        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        detections.sort(key=lambda e: e[2][0])

        if not show_state:
            cv2.imshow('Detection Result %04d' % index , image)
            #detection image 결과 보기
            if cv2.waitKey() & 0xFF == ord('q'):
                break
                
        output_path = "./result/%04d.png" % index
        # data detection 결과가 도출 될 경로

        cv2.imwrite(output_path, image)
        # detection 결과 저장
        index += 1


"""
  darknet image algorithm flow
  main문을 시작으로
  1. parser load
  2. network load(parser data를 활용한 cfg, data,weights 파일 load)
  3. data folder path + image file list load(jpg, jpeg, png 파일이 들어있는 폴더 path)
  4. image file list 내에 파일 개수만큼 반복하여 detection 결과 출력 및 결과 저장
     - image_detection 함수 내에 image_file_path, cfg file, names file, class color 변수를 input
       def image_detection (image_path, network, class_names, class_colors, thresh): 
       (1) width, height load(cfg 정보를 활용)
       (2) yolo에 사용될 width, height 크기에 맞는 빈 이미지 생성
       (3) image_path에 존재하는 image file load
       (4) image를 BGR → RGB 변환
       (5) yolo에 사용될 width, height 크기에 맞게 image resize
       (6) 빈 이미지와 resize image를 사용하여 빈 이미지를 yolo에 사용될 byte 형태로 이미지를 값을 채움
       (7) cfg, class_names, byte형태의 이미지를 토대로 yolo detection 실행하여 detection 값 도출
           ※ detection 값은 list형식으로 return 되고
            list내에 value는 tuple 형식으로 (label, 정확도, (박스 중점 x, 박스 중점 y, 박스 너비, 박스 높이)) 형식으로 출력
       (8) darknet image 메모리 할당 해제(C언어는 memory 할당 및 해제를 직접 해야함)
       (9) detection 값 내에 박스중점(x,y), 박스크기(width,height)를 사용하여 
           원본 이미지 내에 각각 detection box를 draw
       (10) main 함수로 검출된 객체의 box가 그려진 frame과 detection 결과 값을 반환 
        main 함수내에 코드
        image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
        )
  5. 결과값을 토대로 Console 내에 출력
  6. fps 출력
  7. show_state 값에 따라 결과 이미지 확인
     (1) show_state = 0 : 해당 이미지 cv2.imshow
     (2) show_state = 1 : 해당 이미지 결과를 보지 않고 pass
      ※ opencv 결과창에서 esc를 누르면 다음 이미지 결과로 넘어감
  4. - 7. 과정을 파일목록 만큼 loop 하고 종료
       
  각각 함수의 기능과 코드 세부사항은 해당 함수에 작성 
  """


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
