from __future__ import print_function, division
from numpy.core.defchararray import count
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import scipy.io.wavfile as wav
import threading
from multiprocessing import Process, Queue, Array
import pdb
import warnings

import face_alignment
import collections
from imutils.video import VideoStream
import argparse
import imutils
import time
import os, sys
import subprocess
from skimage import io
from os.path import isfile, join
from os import listdir
from tracking_utils import find_q2k, find_outliers
import pickle
import wave
import torch
import torchaudio

import torch.nn as nn
from utils import utils as utils
from torch.utils.data import DataLoader
import torch.nn.utils as torchutils
from torch.autograd import Variable
from utils.logger import Logger
from dcase_util.data import ProbabilityEncoder
import torch.nn.functional as F
from networks.my_TSN import TSN

def extract_frames():

    hasFrame, frame = vs.read()
    resized_crop_img = frame[:,int(240):int(1680),:]
    resized_crop_img = cv2.resize(frame, dsize=(640,480),interpolation=cv2.INTER_LINEAR)
    #files.append(resized_crop_img) 
    Queue_video.put(resized_crop_img)

def extract_audio():

    #frames = []
    data = stream.read(CHUNK_SIZE)
    #frames.append(np.fromstring(data, dtype=np.int16))
    Queue_audio.put(np.fromstring(data, dtype=np.int16))


def start_video(Queue_video, Queue_audio):
    

    while True:
        #video_thread = threading.Thread(target=extract_frames, args= (files,))
        #audio_thread = threading.Thread(target=extract_audio, args= (frames,))

        video_thread = threading.Thread(target=extract_frames)
        audio_thread = threading.Thread(target=extract_audio)

        video_thread.start()    
        audio_thread.start()

        video_thread.join()
        audio_thread.join()


def process_video():
    #torch.multiprocessing.set_start_method('spawn')

    video_counter = 0
    files = []
    frames = []
    video_length = 30

    # import network
    net = utils.import_network(args)

    # init from a saved checkpoint
    if args.pre_train is not '':
        model_name = os.path.join(args.pre_train)

        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(args.pre_train))
            checkpoint = torch.load(args.pre_train)
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.pre_train, checkpoint['epoch']))
        else:
            print('Couldn\'t load model from {}'.format(model_name))
    else:
        print('Training the model from scratch.')



    while True:

        files.append(Queue_video.get())
        frames.append(Queue_audio.get())

        if int(len(files)/video_length) == video_counter + 1:

            video_name = "WIN_video_" + str(video_counter) + ".avi"
            out_path = args.save_video_path + video_name 

            # Face_tracking algorithm
            files_tmp = files[video_length*video_counter:video_length*(video_counter+1)]
            files_tmp_out = []
            out_audio = []
            n_frame = 0
            
            #try:
            for time_dp in range(video_length):    
                frame = files_tmp[time_dp]

                if args.mode == "tracking":
                    ############################ face detect at first frame ############################
                    if n_frame == 0:
                        pred, probs = fa.get_landmarks(frame)
                        if len(probs) > 1:
                            for prob in probs:
                                overlapped_list.append(prob)
                            min_index=overlapped_list.index(max(overlapped_list))
                            pred=[pred[min_index]]
                            overlapped_list=[]
                        
                        pred = np.squeeze(pred)
                        x = pred[:,0]
                        y = pred[:,1]
                        min_x = min(x)
                        min_y = min(y)
                        max_x = max(x)
                        max_y = max(y)

                        height = int((max_y-min_y)/2)
                        width = int((max_x-min_x)/2)
                        standard=max(height,width)

                        box = [int(min_x), int(min_y), int(max_x), int(max_y)]
                        print("first box", box)   

                        box = tuple(box)
                        tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
                        tracker.init(frame, box)
                    else:
                        (success, boxes) = tracker.update(frame)
                        box=[]
                        for i in range(len(boxes)):
                            box.append(int(boxes[i]))
                        box=tuple(box)
                    n_frame += 1
                else:
                    print("NotImplementMode Error")
                    sys.exit()

                (x, y, w, h) = [int(v) for v in box]

                left_boundary=int((h+y)/2)-standard
                right_boundary=int((h+y)/2)+standard
                top_boundary=int((w+x)/2)-standard
                bottom_boundary=int((w+x)/2)+standard


                crop_img = frame[left_boundary:right_boundary,top_boundary:bottom_boundary]
                resized_crop_img=cv2.resize(crop_img, dsize=(224,224),interpolation=cv2.INTER_LINEAR)
                files_tmp_out.append(resized_crop_img) 
                n_frame += 1

            out = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*'DIVX'),
                    fps,
                    size,
                ) 
            print("VIDEO save", out_path)

            for k in range(video_length):
                out.write(files_tmp_out[k])
            out.release()

            #except Exception as ex:
            #    print("Warning: No faces were detected.")
            

            for au in range(video_length*video_counter,video_length*(video_counter+1)):
                out_audio.append(frames[au])

            # save audio
            numpy_data = np.hstack(out_audio)
            audio_name = "WIN_audio_" + str(video_counter) + ".wav"
            a_out_path = args.save_audio_path + audio_name
            wav.write(a_out_path, RATE, numpy_data)
            print("AUDIO save", a_out_path)
            print("---------------------------------------------------------------")

            # create test dataset
            dataset = utils.import_dataset(args)
            test_dataset = dataset(video_name, audio_name, DataDir='/home/nas2/user/jsydshs/VVAD_GIT/data/IIP/DEMO/VIDEO/',audio_DataDir = "/home/nas2/user/jsydshs/VVAD_GIT/data/IIP/DEMO/AUDIO/", timeDepth = args.time_depth, is_train=False)
            batch_size=args.batch_size
            

            # create the data loader
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=False,
                drop_last=True)


            net.eval()
            all_pred = []
            VideoPd={} 

            for i, data in enumerate(test_loader):

                states_test = net.init_hidden(is_train=False)

                if args.arch == 'Video' or args.arch == 'Video_fc'or args.arch == 'Audio':  # single modality
                
                    (input, input_audio) = data # VIDEO & AUDIO MUL JSYDSHS
                    output = net(input.cuda(), states_test, input_audio.cuda())


                else:  # multiple modalities

                    audio, video = data
                    audio_var = Variable(audio.unsqueeze(1)).cuda()
                    video_var = Variable(video.unsqueeze(1)).cuda()

                    output = net(audio_var, video_var, states_test)

                _, predicted = torch.max(output.cpu(), 1)


                #if i % args.print_freq == 0:
                #    print('Test: [{0}/{1}]'.format(i, len(test_loader)))

                for nam in range(batch_size):
                    all_pred.append(predicted[nam].tolist())
                

        #    print("before smoothing: {}".format(all_pred))

                #if len(all_pred) == 15:
            #print("before smoothing: {}".format(all_pred))
            #smoothing for binary score and prediction      
            for i in range(1,len(all_pred)-8):
                if all_pred[i] ==1 and all_pred[i-1]==0:
                    if all_pred[i+1]==0 or all_pred[i+2]==0 or all_pred[i+3]==0 or all_pred[i+4]==0 or all_pred[i+5]==0 or all_pred[i+6]==0 or all_pred[i+7]==0 or all_pred[i+8]==0:
                        all_pred[i]=0
                if all_pred[i] ==0 and all_pred[i-1]==1:
                    if all_pred[i+1]==1 or all_pred[i+2]==1 or all_pred[i+3]==1 or all_pred[i+4]==1 or all_pred[i+5]==1 or all_pred[i+6]==1 or all_pred[i+7]==1 or all_pred[i+8]==1:
                        all_pred[i]=1
   
            print("after smoothing: {}".format(all_pred))
            """
            #pred_video_name=name_stack[name_stack_num].split('.')[0]
            pred_video_name=video_name
            VideoPd[pred_video_name]=np.array(all_pred, dtype=np.int32)
            print(pred_video_name, ": prediction saved")
            all_pred = []
            """
            #name_stack_num +=1

                #with open('VideoPd_list_seyeong.txt', 'wb') as f:
                #    pickle.dump(VideoPd, f)     

        #        print('Test finished.')
            video_counter += 1


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    #warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=2, help='test batch size')
    parser.add_argument('--time_depth', type=int, default=15, help='number of time frames in each video\audio sample')
    parser.add_argument('--workers', type=int, default=40, help='num workers for data loading')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay factor')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum factor')
    parser.add_argument('--save_freq', type=int, default=1, help='freq of saving the model')
    parser.add_argument('--print_freq', type=int, default=50, help='freq of printing stats')
    parser.add_argument('--seed', type=int, default=44974274, help='random seed')
    parser.add_argument('--lstm_layers', type=int, default=2, help='number of lstm layers in the model')
    parser.add_argument('--lstm_hidden_size', type=int, default=1024, help='number of neurons in each lstm layer in the model')
    parser.add_argument('--use_mcb', action='store_true', help='wether to use MCB or concat')
    parser.add_argument('--mcb_output_size', type=int, default=1024, help='the size of the MCB outputl')
    parser.add_argument('--debug', action='store_true', help='print debug outputs')
    parser.add_argument('--freeze_layers', action='store_true', help='wether to freeze the first layers of the model')
    parser.add_argument('--arch', type=str, default='Video', choices=['Audio', 'Video', 'AV'], help='which modality to train - Video\Audio\Multimodal')
    #parser.add_argument('--pre_train', type=str, default='/home/nas/user/jsydshs/VVAD_GIT/VVAD_MUL_MOB_TSN_AT_END/saved_models/Video/acc_95.593_epoch_015_arch_Video.pkl', help='path to a pre-trained network')
    parser.add_argument('--pre_train', type=str, default='/home/nas/user/jsydshs/VVAD_GIT/VVAD_MUL_MOB_TSN_AT_END_R/saved_models/Video/acc_90.139_epoch_019_arch_Video.pkl', help='path to a pre-trained network')
    parser.add_argument('--demo', type=str, default='demo', help='where are we gonna get VideoDataset from? demo_dataset or dataset?')
    parser.add_argument("-v", "--video_path",default='/home/nas2/user/jsydshs/VVAD_GIT/data/IIP/DEMO/Crop_tmp/', type=str, help="path to input video file")
    parser.add_argument("-sv", "--save_video_path", type=str,default='/home/nas2/user/jsydshs/VVAD_GIT/data/IIP/DEMO/VIDEO/', help="path to output video file")
    parser.add_argument("-sa", "--save_audio_path", type=str,default='/home/nas2/user/jsydshs/VVAD_GIT/data/IIP/DEMO/AUDIO/', help="path to output audio file")
    parser.add_argument("-m", "--mode", type=str, default="tracking", help="select detect, tracking or both", )
    parser.add_argument("-t", "--tracker", type=str, default="medianflow", help="OpenCV object tracker type")
    args = parser.parse_args()

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create,
        "goturn":cv2.TrackerGOTURN_create
    }

    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False, face_detector='sfd')
    fa_probs_threshold  = 0.95

    RATE = 44100
    CHUNK_SIZE = 1470

    # audio setting
    pypy = pyaudio.PyAudio()
    stream = pypy.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    #frames = []  
    
    # video setting
    vs = cv2.VideoCapture(0) 
    fps = 30.0
    size=(224,224)  

    while True:
        Queue_video = Queue()
        Queue_audio = Queue()

        streaming = threading.Thread(target=start_video, args=(Queue_video,Queue_audio,))
        processing = threading.Thread(target=process_video)

        streaming.start()
        processing.start()

        streaming.join()
        processing.join()



        