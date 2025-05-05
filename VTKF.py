# -*- coding: utf-8 -*-
import os,sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label import Label, lwrite
from os.path import splitext, basename, isdir
from os import makedirs
from src.utils import crop_region, image_files_from_folder, im2single
from darknet.python.darknet import detect, nparray_to_image
from src.drawing_utils import draw_label, draw_losangle, draw_losangle_new
from src.keras_utils import load_model, detect_lp
from src.label import Shape, Label
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from filterpy.kalman import KalmanFilter

def do_mosaic(frame, x, y, w, h, neighbor=5):
	"""
	:param frame: opencv frame
    :param int x : top-left
    :param int y: 
    :param int w: weight
    :param int h: high
    :param int neighbor: 
	"""
	fh, fw = frame.shape[0], frame.shape[1]
	if (y + h > fh) or (x + w > fw):
		return
	for i in range(0, h - neighbor, neighbor):
		for j in range(0, w - neighbor, neighbor):
			rect = [j + x, i + y, neighbor, neighbor]
			color = frame[i + y][j + x].tolist()
			left_up = (rect[0], rect[1])
			right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)
			cv2.rectangle(frame, left_up, right_down, color, -1)

def isInside(x1, y1, x4, y4, x, y):
	"""
	:param x1: top-left
	:param y1:
	:param x4: right-down
	:param y4:
	:param x: point
	:param y:
	:return:
	"""
	if x < x1 or x > x4 or y < y1 or y > y4:
		return 0
	return 1


class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    # define constant velocity model
    self.kf = KalmanFilter(dim_x=8, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],  [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0]])

    self.kf.R[2:,2:] *= 0.001
    self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

	# initialization
    # self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.kf.x[:4] = np.array(bbox).reshape((4,1))
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    # self.kf.update(convert_bbox_to_z(bbox))
    self.kf.update(np.array(bbox).reshape((4,1)))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    # self.history.append(convert_x_to_bbox(self.kf.x))
    self.history.append(np.array(self.kf.x[:4]).reshape((1,4)))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    # return convert_x_to_bbox(self.kf.x)
    return self.kf.x


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    # scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))


def convert_bbox_to_r(bbox):
	"""
	Get relative position.
	:param bbox: [bbox of car(top-left, right-down), bbox of plate(top-left, right-down)]
	:return: relative position
	"""
	car_w = bbox[2] - bbox[0]
	car_h = bbox[3] - bbox[1]
	plate_w = bbox[6] - bbox[4]
	plate_h = bbox[7] - bbox[5]
	r_tl_x = float(bbox[4] - bbox[0])/car_w
	r_tl_y = float(bbox[5] - bbox[1])/car_h
	r_w = float(plate_w)/car_w
	r_h = float(plate_h)/car_h
	return np.array([r_tl_x, r_tl_y, r_w, r_h]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))



def getCarsDir(fileName):
	"""
	Get the vehicle position from the tracking result.
	"""
	carDir = {}
	fr = open(fileName)
	for line in fr.readlines():  
		lineArr = line.strip().split(', ')
		if int(lineArr[0]) not in carDir.keys():
			if lineArr[1] == "car" or lineArr[1] == "truck" or lineArr[1] == "bus" or lineArr[1] == "2" or lineArr[1] == "5" or lineArr[1] == "7":
				if int(lineArr[7])-int(lineArr[3])>0 and int(lineArr[8])-int(lineArr[4])>0:
					index_car = {}
					carlocal = {}
					carlocal["carlocal"] = [int(lineArr[3]),int(lineArr[4]),int(lineArr[5]),int(lineArr[6]),int(lineArr[7]),int(lineArr[8]),int(lineArr[9]),int(lineArr[10]),float(lineArr[11])]
					carlocal["platelocal"] = []
					index_car[int(lineArr[2])] = carlocal
					carDir[int(lineArr[0])] = index_car
		else:
			if lineArr[1] == "car" or lineArr[1] == "truck" or lineArr[1] == "bus" or lineArr[1] == "2" or lineArr[1] == "5" or lineArr[1] == "7":
				if int(lineArr[7]) - int(lineArr[3]) > 0 and int(lineArr[8]) - int(lineArr[4]) > 0:  
					carlocal = {}
					carlocal["carlocal"] = [int(lineArr[3]), int(lineArr[4]), int(lineArr[5]), int(lineArr[6]),int(lineArr[7]), int(lineArr[8]), int(lineArr[9]), int(lineArr[10]),float(lineArr[11])]
					carlocal["platelocal"] = []
					carDir[int(lineArr[0])][int(lineArr[2])] = carlocal
	# carDir = find_mischeck_car(carDir, 1)
	return carDir


def find_mischeck_car(carDir,tiaozhen):
	"""
	Remove the misdetected vehicle information and fill in the removed vehicle information based on interpolation.
	"""
	for frame_id in carDir:
		if frame_id - tiaozhen >0:
			for car_id in carDir[frame_id]:
				if car_id in carDir[frame_id - tiaozhen].keys() and len(carDir[frame_id - tiaozhen][car_id]["carlocal"]) > 0:
					
					pre_center_x = int((carDir[frame_id - tiaozhen][car_id]["carlocal"][0] + carDir[frame_id - tiaozhen][car_id]["carlocal"][4]) / 2)
					pre_center_y = int((carDir[frame_id - tiaozhen][car_id]["carlocal"][1] + carDir[frame_id - tiaozhen][car_id]["carlocal"][5]) / 2)
					
					index_center_x = int((carDir[frame_id][car_id]["carlocal"][0] + carDir[frame_id][car_id]["carlocal"][4]) / 2)
					index_center_y = int((carDir[frame_id][car_id]["carlocal"][1] + carDir[frame_id][car_id]["carlocal"][5]) / 2)
					
					move_dis = ((pre_center_x - index_center_x)**2 + (pre_center_y - index_center_y)**2)**0.5
					if move_dis > 180:
						
						if car_id in carDir[frame_id + tiaozhen].keys() and frame_id + tiaozhen <= len(carDir) * tiaozhen:
							carDir[frame_id][car_id]["carlocal"][0] = int((carDir[frame_id - tiaozhen][car_id]["carlocal"][0] + carDir[frame_id + tiaozhen][car_id]["carlocal"][0]) /2)
							carDir[frame_id][car_id]["carlocal"][1] = int((carDir[frame_id - tiaozhen][car_id]["carlocal"][1] + carDir[frame_id + tiaozhen][car_id]["carlocal"][1]) /2)
							carDir[frame_id][car_id]["carlocal"][2] = int((carDir[frame_id - tiaozhen][car_id]["carlocal"][2] + carDir[frame_id + tiaozhen][car_id]["carlocal"][2]) /2)
							carDir[frame_id][car_id]["carlocal"][3] = int((carDir[frame_id - tiaozhen][car_id]["carlocal"][3] + carDir[frame_id + tiaozhen][car_id]["carlocal"][3]) /2)
							carDir[frame_id][car_id]["carlocal"][4] = int((carDir[frame_id - tiaozhen][car_id]["carlocal"][4] + carDir[frame_id + tiaozhen][car_id]["carlocal"][4]) /2)
							carDir[frame_id][car_id]["carlocal"][5] = int((carDir[frame_id - tiaozhen][car_id]["carlocal"][5] + carDir[frame_id + tiaozhen][car_id]["carlocal"][5]) /2)
							carDir[frame_id][car_id]["carlocal"][6] = int((carDir[frame_id - tiaozhen][car_id]["carlocal"][6] + carDir[frame_id + tiaozhen][car_id]["carlocal"][6]) /2)
							carDir[frame_id][car_id]["carlocal"][7] = int((carDir[frame_id - tiaozhen][car_id]["carlocal"][7] + carDir[frame_id + tiaozhen][car_id]["carlocal"][7]) /2)
							carDir[frame_id][car_id]["carlocal"][8] = (carDir[frame_id - tiaozhen][car_id]["carlocal"][8] + carDir[frame_id + tiaozhen][car_id]["carlocal"][8]) /2
						else:
							carDir[frame_id][car_id]["carlocal"][0] = carDir[frame_id - tiaozhen][car_id]["carlocal"][0]
							carDir[frame_id][car_id]["carlocal"][1] = carDir[frame_id - tiaozhen][car_id]["carlocal"][1]
							carDir[frame_id][car_id]["carlocal"][2] = carDir[frame_id - tiaozhen][car_id]["carlocal"][2]
							carDir[frame_id][car_id]["carlocal"][3] = carDir[frame_id - tiaozhen][car_id]["carlocal"][3]
							carDir[frame_id][car_id]["carlocal"][4] = carDir[frame_id - tiaozhen][car_id]["carlocal"][4]
							carDir[frame_id][car_id]["carlocal"][5] = carDir[frame_id - tiaozhen][car_id]["carlocal"][5]
							carDir[frame_id][car_id]["carlocal"][6] = carDir[frame_id - tiaozhen][car_id]["carlocal"][6]
							carDir[frame_id][car_id]["carlocal"][7] = carDir[frame_id - tiaozhen][car_id]["carlocal"][7]
							carDir[frame_id][car_id]["carlocal"][8] = carDir[frame_id - tiaozhen][car_id]["carlocal"][8]

	return carDir


def get_test_sampling(testList):
	"""
	Generate binary classifier test samples.
	"""
	car_w = testList[4] - testList[0]
	car_h = testList[5] - testList[1]
	car_area = car_w * car_h
	plate_w = testList[13] - testList[9]
	plate_h = testList[14] - testList[10]
	plate_area = plate_w * plate_h
	plate_r = float(plate_w)/plate_h
	reslist = []
	reslist.extend(testList[:9])
	reslist.extend([car_area])
	reslist.extend(testList[9:])
	reslist.extend([plate_r,plate_area])
	return reslist



def loadDataSet(fileName):
	"""
	Get training data and corresponding labels.
	"""
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():  
		print(line)
		lineArr = line.strip().split(',')
		print(lineArr)
		dataMat.append([int(lineArr[0]),int(lineArr[1]),int(lineArr[2]),int(lineArr[3]),int(lineArr[4]),int(lineArr[5]),int(lineArr[6]),int(lineArr[7]),float(lineArr[8]),
						int(float(lineArr[9])),int(float(lineArr[10])),int(float(lineArr[11])),int(float(lineArr[12])),int(float(lineArr[13])),int(float(lineArr[14])),int(float(lineArr[15])),int(float(lineArr[16])),int(float(lineArr[17])),float(lineArr[18]),float(lineArr[19]),int(float(lineArr[20]))])  
		labelMat.append(int(lineArr[21]))  
	return dataMat, labelMat



def plateAdaBoost(testList):
	"""
	Two-classification to remove false license plate detection.
	"""
	trainList, trainLabel = loadDataSet("traindata1.txt")
	# print(trainList)
	# print(trainLabel)
	# AdaBoost
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=10)
	bdt.fit(trainList, trainLabel)
	predictions = bdt.predict(testList)
	# print("AdaBoost:", predictions)

	return predictions



def find_leakdec_plate(frameDir, frame_index, car_id, maxfind,tiaozhen):
	"""
	Find the missed license plate and add the license plate information.
	"""
	pre_have = 0
	pre_frame={}
	
	back_have = 0
	back_frame={}

	for i in range(maxfind):
		pre_index = frame_index-(i+1)*tiaozhen
		
		if pre_index>=tiaozhen and car_id in frameDir[pre_index].keys() and len(frameDir[pre_index][car_id]["platelocal"])!=0:
			pre_have =1
			pre_frame = frameDir[pre_index][car_id]
			break

	for i in range(maxfind):
		back_index = frame_index+(i+1)*tiaozhen
		
		if back_index<=len(frameDir)*tiaozhen and car_id in frameDir[back_index].keys() and len(frameDir[back_index][car_id]["platelocal"])!=0:
			back_have =1
			back_frame = frameDir[back_index][car_id]
			break

	if pre_have==1 and back_have==1:
		pre_car_width = pre_frame["carlocal"][4] - pre_frame["carlocal"][0]
		pre_car_height = pre_frame["carlocal"][5] - pre_frame["carlocal"][1]

		pre_left_x = pre_frame["platelocal"][0]
		pre_left_y = pre_frame["platelocal"][1]
		pre_plate_width = pre_frame["platelocal"][4] - pre_frame["platelocal"][0]
		pre_plate_height = pre_frame["platelocal"][5] - pre_frame["platelocal"][1]

		pre_x_rel = (pre_frame["platelocal"][0]-pre_frame["carlocal"][0])/pre_car_width
		pre_y_rel = (pre_frame["platelocal"][1]-pre_frame["carlocal"][1])/pre_car_height

		pre_w_rel = pre_plate_width/pre_car_width
		pre_h_rel = pre_plate_height/pre_car_height

		pre_score = pre_frame["platelocal"][8]

		back_car_width = back_frame["carlocal"][4] - back_frame["carlocal"][0]
		back_car_height = back_frame["carlocal"][5] - back_frame["carlocal"][1]

		back_left_x = back_frame["platelocal"][0]
		back_left_y = back_frame["platelocal"][1]
		back_plate_width = back_frame["platelocal"][4] - back_frame["platelocal"][0]
		back_plate_height = back_frame["platelocal"][5] - back_frame["platelocal"][1]

		back_x_rel = (back_frame["platelocal"][0] - back_frame["carlocal"][0]) / back_car_width
		back_y_rel = (back_frame["platelocal"][1] - back_frame["carlocal"][1]) / back_car_height

		back_w_rel = back_plate_width / back_car_width
		back_h_rel = back_plate_height / back_car_height

		back_score = back_frame["platelocal"][8]

		leak_dec_num =  int((back_index - frame_index)/tiaozhen)
		
		for j in range(leak_dec_num):
			if car_id in frameDir[frame_index+j*tiaozhen].keys():
				leakdec_frame = frameDir[frame_index+j*tiaozhen][car_id]
				leakdec_car_width = leakdec_frame["carlocal"][4] - leakdec_frame["carlocal"][0]
				leakdec_car_height = leakdec_frame["carlocal"][5] - leakdec_frame["carlocal"][1]
				leakdec_plate_lx = leakdec_frame["carlocal"][0] + leakdec_car_width * ((pre_x_rel + back_x_rel)/2)
				leakdec_plate_ly = leakdec_frame["carlocal"][1] + leakdec_car_height * ((pre_y_rel + back_y_rel)/2)
				leakdec_plate_w = leakdec_car_width * ((pre_w_rel + back_w_rel)/2)
				leakdec_plate_h = leakdec_car_height * ((pre_h_rel + back_h_rel)/2)
				leakdec_plate_s = (pre_score + back_score)/2

				if leakdec_plate_w<=0 or leakdec_plate_h<=0:
					break
				else:
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + 0.5),int(leakdec_plate_ly + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + leakdec_plate_w + 0.5),int(leakdec_plate_ly + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + leakdec_plate_w + 0.5),int(leakdec_plate_ly + leakdec_plate_h + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + 0.5),int(leakdec_plate_ly + leakdec_plate_h + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([leakdec_plate_s])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([2])

	elif pre_have == 1 and back_have == 0:  
		pre_car_width = pre_frame["carlocal"][4] - pre_frame["carlocal"][0]
		pre_car_height = pre_frame["carlocal"][5] - pre_frame["carlocal"][1]

		pre_left_x = pre_frame["platelocal"][0]
		pre_left_y = pre_frame["platelocal"][1]
		pre_plate_width = pre_frame["platelocal"][4] - pre_frame["platelocal"][0]
		pre_plate_height = pre_frame["platelocal"][5] - pre_frame["platelocal"][1]

		pre_x_rel = (pre_frame["platelocal"][0] - pre_frame["carlocal"][0]) / pre_car_width
		pre_y_rel = (pre_frame["platelocal"][1] - pre_frame["carlocal"][1]) / pre_car_height

		pre_w_rel = pre_plate_width / pre_car_width
		pre_h_rel = pre_plate_height / pre_car_height

		pre_score = pre_frame["platelocal"][8]

		leak_dec_num = int((frame_index - pre_index)/tiaozhen)
		for j in range(leak_dec_num):
			if car_id in frameDir[frame_index - j*tiaozhen].keys():
				leakdec_frame = frameDir[frame_index - j*tiaozhen][car_id]
				leakdec_car_width = leakdec_frame["carlocal"][4] - leakdec_frame["carlocal"][0]
				leakdec_car_height = leakdec_frame["carlocal"][5] - leakdec_frame["carlocal"][1]
				leakdec_plate_lx = leakdec_frame["carlocal"][0] + leakdec_car_width * pre_x_rel
				leakdec_plate_ly = leakdec_frame["carlocal"][1] + leakdec_car_height * pre_y_rel
				leakdec_plate_w = leakdec_car_width * pre_w_rel
				leakdec_plate_h = leakdec_car_height * pre_h_rel
				leakdec_plate_s = pre_score

				if leakdec_plate_w <= 0 or leakdec_plate_h <= 0:
					break
				else:
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + 0.5), int(leakdec_plate_ly + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + leakdec_plate_w + 0.5), int(leakdec_plate_ly + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + leakdec_plate_w + 0.5), int(leakdec_plate_ly + leakdec_plate_h + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + 0.5), int(leakdec_plate_ly + leakdec_plate_h + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([leakdec_plate_s])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([2])

	elif pre_have == 0 and back_have == 1:  
		back_car_width = back_frame["carlocal"][4] - back_frame["carlocal"][0]
		back_car_height = back_frame["carlocal"][5] - back_frame["carlocal"][1]

		back_left_x = back_frame["platelocal"][0]
		back_left_y = back_frame["platelocal"][1]
		back_plate_width = back_frame["platelocal"][4] - back_frame["platelocal"][0]
		back_plate_height = back_frame["platelocal"][5] - back_frame["platelocal"][1]

		back_x_rel = (back_frame["platelocal"][0] - back_frame["carlocal"][0]) / back_car_width
		back_y_rel = (back_frame["platelocal"][1] - back_frame["carlocal"][1]) / back_car_height

		back_w_rel = back_plate_width / back_car_width
		back_h_rel = back_plate_height / back_car_height

		back_score = back_frame["platelocal"][8]

		leak_dec_num = int((back_index - frame_index)/tiaozhen)
		for j in range(leak_dec_num):
			if car_id in frameDir[frame_index + j*tiaozhen].keys():
				leakdec_frame = frameDir[frame_index + j*tiaozhen][car_id]
				leakdec_car_width = leakdec_frame["carlocal"][4] - leakdec_frame["carlocal"][0]
				leakdec_car_height = leakdec_frame["carlocal"][5] - leakdec_frame["carlocal"][1]
				leakdec_plate_lx = leakdec_frame["carlocal"][0] + leakdec_car_width * back_x_rel
				leakdec_plate_ly = leakdec_frame["carlocal"][1] + leakdec_car_height * back_y_rel
				leakdec_plate_w = leakdec_car_width * back_w_rel
				leakdec_plate_h = leakdec_car_height * back_h_rel
				leakdec_plate_s = back_score

				if leakdec_plate_w <= 0 or leakdec_plate_h <= 0:
					break
				else:
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + 0.5), int(leakdec_plate_ly + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + leakdec_plate_w + 0.5), int(leakdec_plate_ly + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + leakdec_plate_w + 0.5), int(leakdec_plate_ly + leakdec_plate_h + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([int(leakdec_plate_lx + 0.5), int(leakdec_plate_ly + leakdec_plate_h + 0.5)])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([leakdec_plate_s])
					frameDir[frame_index + j*tiaozhen][car_id]["platelocal"].extend([2])


def convert_kalman_data(carlist,platelist):
	"""
	Convert data formats
	"""
	p_tl_x = min(platelist[0], platelist[2], platelist[4], platelist[6])
	p_tl_y = min(platelist[1], platelist[3], platelist[5], platelist[7])
	p_br_x = max(platelist[0], platelist[2], platelist[4], platelist[6])
	p_br_y = max(platelist[1], platelist[3], platelist[5], platelist[7])
	platelist = []
	platelist.append(p_tl_x)
	platelist.append(p_tl_y)
	platelist.append(p_br_x)
	platelist.append(p_br_y)
	car_w = carlist[4] - carlist[0]
	car_h = carlist[5] - carlist[1]
	plate_w = platelist[2] - platelist[0]
	plate_h = platelist[3] - platelist[1]
	r_tl_x = float(platelist[0] - carlist[0]) / car_w
	r_tl_y = float(platelist[1] - carlist[1]) / car_h
	r_w = float(plate_w) / car_w
	r_h = float(plate_h) / car_h
	det = [r_tl_x, r_tl_y, r_w, r_h]
	return det



def get_platelocal_from_r(carlist,r_list):
	"""
	Get the license plate coordinates based on the relative position and vehicle coordinates
	"""
	car_w = carlist[4] - carlist[0]
	car_h = carlist[5] - carlist[1]
	plate_tl_x = carlist[0] + car_w * r_list[0]
	plate_tl_y = carlist[1] + car_h * r_list[1]
	plate_w = car_w * r_list[2]
	plate_h = car_h * r_list[3]
	platelist = []
	platelist.extend([plate_tl_x,plate_tl_y])
	platelist.extend([plate_tl_x + plate_w,plate_tl_y])
	platelist.extend([plate_tl_x + plate_w,plate_tl_y + plate_h])
	platelist.extend([plate_tl_x,plate_tl_y + plate_h])
	return platelist



def enlarge_platelocal(platelocal,enlarge_r):
	c_x = (platelocal[0] + platelocal[4])/2
	c_y = (platelocal[1] + platelocal[5])/2
	plate_w = platelocal[4] - platelocal[0]
	plate_h = platelocal[5] - platelocal[1]
	enlarge_platelocal = [0,0,0,0,0,0,0,0,0]
	enlarge_platelocal[0] = c_x - plate_w / 2
	enlarge_platelocal[1] = c_y - plate_h / 2 * enlarge_r
	enlarge_platelocal[4] = c_x + plate_w / 2
	enlarge_platelocal[5] = c_y + plate_h / 2 * enlarge_r
	if enlarge_platelocal[0] < 0:
		enlarge_platelocal[0] = 0
	if enlarge_platelocal[1] < 0:
		enlarge_platelocal[1] = 0
	enlarge_platelocal[2] = enlarge_platelocal[4]
	enlarge_platelocal[3] = enlarge_platelocal[1]
	enlarge_platelocal[6] = enlarge_platelocal[0]
	enlarge_platelocal[7] = enlarge_platelocal[5]
	enlarge_platelocal[8] = platelocal[8]
	return enlarge_platelocal



def get_leakplate_through_bidirectional_kalman(frameDir, frame_index, car_id, maxfind, tiaozhen):
	"""
	Bidirectional Kalman filter estimation of missed license plates
	"""
	pre_frame_list = []
	back_frame_list = []

	frame_keys = list(frameDir.keys())
	key_index = frame_keys.index(frame_index)

	for i in range(maxfind):
		pre_key_index = key_index - (i + 1)
		if pre_key_index >= 0:
			pre_index = frame_keys[pre_key_index]
			if pre_index >= tiaozhen and car_id in frameDir[pre_index].keys() and len(frameDir[pre_index][car_id]["platelocal"]) != 0:
				pre_frame_list.append(pre_index)

	for i in range(maxfind):
		back_key_index = key_index + (i + 1)
		if back_key_index <= len(frame_keys)-1:
			back_index = frame_keys[back_key_index]
			if back_index <= len(frameDir) * tiaozhen and car_id in frameDir[back_index].keys() and len(frameDir[back_index][car_id]["platelocal"]) != 0:
				back_frame_list.append(back_index)

	if len(pre_frame_list) > 0 and len(back_frame_list) > 0:  
		pre_frame_list.reverse()

		platelist = frameDir[pre_frame_list[0]][car_id]["platelocal"]
		carlist = frameDir[pre_frame_list[0]][car_id]["carlocal"]
		det = convert_kalman_data(carlist,platelist)

		tracker = KalmanBoxTracker(np.array(det))

		bejin = frame_keys.index(pre_frame_list[0])
		end = frame_keys.index(back_frame_list[-1])
		for index in range(bejin+1,end+1):
			i = frame_keys[index]
			pos = tracker.predict()[0]

			if car_id in frameDir[i].keys():
				if len(frameDir[i][car_id]["platelocal"])>0:
					detection = convert_kalman_data(frameDir[i][car_id]["carlocal"],frameDir[i][car_id]["platelocal"])
					tracker.update(np.array(detection))
				elif len(frameDir[i][car_id]["platelocal"])==0:
					frameDir[i][car_id]["pre_kalman_platelocal"] = get_platelocal_from_r(frameDir[i][car_id]["carlocal"],[pos[0],pos[1],pos[2],pos[3]])
					# if len(frameDir[frame_keys[index-1]][car_id]["platelocal"])==0:
					# 	index_score = frameDir[frame_keys[index - 1]][car_id]["pre_kalman_platelocal"][8]
					# else:
					# 	index_score = frameDir[frame_keys[index-1]][car_id]["platelocal"][8]
					frameDir[i][car_id]["pre_kalman_platelocal"].append(0.85)
					detection = convert_kalman_data(frameDir[i][car_id]["carlocal"], frameDir[i][car_id]["pre_kalman_platelocal"])
					tracker.update(np.array(detection))


		platelist = frameDir[back_frame_list[-1]][car_id]["platelocal"]
		carlist = frameDir[back_frame_list[-1]][car_id]["carlocal"]
		det = convert_kalman_data(carlist, platelist)

		tracker = KalmanBoxTracker(np.array(det))

		for index in range(end-1, bejin-1, -1):
			i = frame_keys[index]
			pos = tracker.predict()[0]

			if car_id in frameDir[i].keys():
				if len(frameDir[i][car_id]["platelocal"]) > 0:
					detection = convert_kalman_data(frameDir[i][car_id]["carlocal"], frameDir[i][car_id]["platelocal"])
					tracker.update(np.array(detection))
				elif len(frameDir[i][car_id]["platelocal"]) == 0:
					frameDir[i][car_id]["back_kalman_platelocal"] = get_platelocal_from_r(frameDir[i][car_id]["carlocal"], [pos[0], pos[1], pos[2], pos[3]])
					# if len(frameDir[frame_keys[index + 1]][car_id]["platelocal"])==0:
					# 	index_score = frameDir[frame_keys[index + 1]][car_id]["back_kalman_platelocal"][8]
					# else:
					# 	index_score = frameDir[frame_keys[index + 1]][car_id]["platelocal"][8]
					frameDir[i][car_id]["back_kalman_platelocal"].append(0.85)

					pre_platelocal = frameDir[i][car_id]["pre_kalman_platelocal"]
					back_platelocal = frameDir[i][car_id]["back_kalman_platelocal"]
					predict_plate = [int((pre + back) / 2) for pre, back in zip(pre_platelocal, back_platelocal)]

					frameDir[i][car_id]["platelocal"] = enlarge_platelocal(predict_plate, 1)
					frameDir[i][car_id]["platelocal"].extend([2])

					detection = convert_kalman_data(frameDir[i][car_id]["carlocal"], frameDir[i][car_id]["back_kalman_platelocal"])

					tracker.update(np.array(detection))

	elif len(pre_frame_list) > 0 and len(back_frame_list) == 0:  
		pre_frame_list.reverse()

		platelist = frameDir[pre_frame_list[0]][car_id]["platelocal"]
		carlist = frameDir[pre_frame_list[0]][car_id]["carlocal"]
		det = convert_kalman_data(carlist, platelist)

		tracker = KalmanBoxTracker(np.array(det))

		bejin = frame_keys.index(pre_frame_list[0])
		for index in range(bejin+1, key_index+1):
			i = frame_keys[index]
			pos = tracker.predict()[0]

			if car_id in frameDir[i].keys():
				if len(frameDir[i][car_id]["platelocal"]) > 0:

					detection = convert_kalman_data(frameDir[i][car_id]["carlocal"], frameDir[i][car_id]["platelocal"])

					tracker.update(np.array(detection))
				elif len(frameDir[i][car_id]["platelocal"]) == 0:
					predict_plate = get_platelocal_from_r(frameDir[i][car_id]["carlocal"], [pos[0], pos[1], pos[2], pos[3]])
					predict_plate.append(0.85)
					frameDir[i][car_id]["platelocal"] = enlarge_platelocal(predict_plate, 1)
					frameDir[i][car_id]["platelocal"].extend([2])
					detection = convert_kalman_data(frameDir[i][car_id]["carlocal"], frameDir[i][car_id]["platelocal"])

					tracker.update(np.array(detection))


	elif len(pre_frame_list) == 0 and len(back_frame_list) > 0:  
		platelist = frameDir[back_frame_list[-1]][car_id]["platelocal"]
		carlist = frameDir[back_frame_list[-1]][car_id]["carlocal"]
		det = convert_kalman_data(carlist, platelist)

		tracker = KalmanBoxTracker(np.array(det))

		end = frame_keys.index(back_frame_list[-1])
		for index in range(end-1, key_index-1, (-1)):
			i = frame_keys[index]
			pos = tracker.predict()[0]

			if car_id in frameDir[i].keys():
				if len(frameDir[i][car_id]["platelocal"]) > 0:
					detection = convert_kalman_data(frameDir[i][car_id]["carlocal"], frameDir[i][car_id]["platelocal"])

					tracker.update(np.array(detection))
				elif len(frameDir[i][car_id]["platelocal"]) == 0:
					predict_plate = get_platelocal_from_r(frameDir[i][car_id]["carlocal"], [pos[0], pos[1], pos[2], pos[3]])
					# index_score = frameDir[frame_keys[index + 1]][car_id]["platelocal"][8]
					predict_plate.append(0.85)
					frameDir[i][car_id]["platelocal"] = enlarge_platelocal(predict_plate, 1)
					frameDir[i][car_id]["platelocal"].extend([2])

					detection = convert_kalman_data(frameDir[i][car_id]["carlocal"], frameDir[i][car_id]["platelocal"])

					tracker.update(np.array(detection))



def drawplate(frameDir,input_video_path,video_name,output_video_path):
	maxfind = 15
	frame_index = -1
	car_id = -1
	tiaozhen = 1

	# fp = open(os.path.join(output_video_path+"/before_dir",video_name.replace('.mp4', '_before_dir.txt')), "w")
	# fp.write(str(frameDir))
	# fp.close()

	print("predict leakplate start!!!")
	for frameid in frameDir:
		for carid in frameDir[frameid]:
			if len(frameDir[frameid][carid]["platelocal"]) == 0:
				car_id = carid
				frame_index = int(frameid)
				# find_leakdec_plate(frameDir, frame_index, car_id, maxfind, tiaozhen)
				get_leakplate_through_bidirectional_kalman(frameDir, frame_index, car_id, maxfind, tiaozhen)

	print("predict leakplate end!!!")

	# fp = open(os.path.join(output_video_path+"/final_dir",video_name.replace('.mp4', '_final_dir.txt')), "w")
	# fp.write(str(frameDir))
	# fp.close()

	vid = cv2.VideoCapture(os.path.join(input_video_path,video_name))
	out_file = os.path.join(output_video_path,video_name.replace('.mp4', '_final_result.mp4'))
	fps = vid.get(cv2.CAP_PROP_FPS)  
	vid_writer = cv2.VideoWriter(out_file, \
								 cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), int(fps / tiaozhen), \
								 (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), \
								  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
	# vid_writer = cv2.VideoWriter(out_file, \
	# 							 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(fps / tiaozhen), \
	# 							 (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), \
	# 							  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
	tiaozhenCountFinal = 1
	timeFinal = tiaozhen  
	print("get labeltxt start!!!")
	while True:
		return_value, arr = vid.read()
		img_w = arr.shape[1]
		img_h = arr.shape[0]
		if (tiaozhenCountFinal % timeFinal == 0) and tiaozhenCountFinal in frameDir.keys():  
			for car in frameDir[tiaozhenCountFinal]:
				# print("car:", frameDir[tiaozhenCountFinal][car])
				cv2.rectangle(arr, (int(frameDir[tiaozhenCountFinal][car]["carlocal"][0]),
									int(frameDir[tiaozhenCountFinal][car]["carlocal"][1])),
							  (int(frameDir[tiaozhenCountFinal][car]["carlocal"][4]),
							   int(frameDir[tiaozhenCountFinal][car]["carlocal"][5])), (0, 255, 0), 2)
				cv2.putText(arr, str(car), (int(frameDir[tiaozhenCountFinal][car]["carlocal"][0]),
											int(frameDir[tiaozhenCountFinal][car]["carlocal"][1])),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
				if len(frameDir[tiaozhenCountFinal][car]["platelocal"]) > 0:
					platelocal = frameDir[tiaozhenCountFinal][car]["platelocal"]
					carlocal = frameDir[tiaozhenCountFinal][car]["carlocal"]
					tl_point = isInside(carlocal[0],carlocal[1],carlocal[4],carlocal[5]+20,int(platelocal[0]), int(platelocal[1]))
					br_point = isInside(carlocal[0],carlocal[1],carlocal[4],carlocal[5]+20,int(platelocal[4]),int(platelocal[5]))
					if tl_point and br_point:
						# cv2.line(arr, (int(platelocal[0]),int(platelocal[1])), (int(platelocal[2]),int(platelocal[3])), (0, 255, 0), 3)
						# cv2.line(arr, (int(platelocal[2]),int(platelocal[3])), (int(platelocal[4]),int(platelocal[5])), (0, 255, 0), 3)
						# cv2.line(arr, (int(platelocal[4]),int(platelocal[5])), (int(platelocal[6]),int(platelocal[7])), (0, 255, 0), 3)
						# cv2.line(arr, (int(platelocal[6]),int(platelocal[7])), (int(platelocal[0]),int(platelocal[1])), (0, 255, 0), 3)

						p_tl_x = min(platelocal[0], platelocal[2], platelocal[4], platelocal[6])
						p_tl_y = min(platelocal[1], platelocal[3], platelocal[5], platelocal[7])
						p_br_x = max(platelocal[0], platelocal[2], platelocal[4], platelocal[6])
						p_br_y = max(platelocal[1], platelocal[3], platelocal[5], platelocal[7])
						cv2.rectangle(arr, (int(p_tl_x),int(p_tl_y)),(int(p_br_x),int(p_br_y)), (0, 255, 0), 2)

						cv2.putText(arr, str(frameDir[tiaozhenCountFinal][car]["platelocal"][9]), (int(frameDir[tiaozhenCountFinal][car]["platelocal"][0]),
													int(frameDir[tiaozhenCountFinal][car]["platelocal"][1])),
									cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

						if (tiaozhenCountFinal % 30 == 0):
							if tiaozhenCountFinal > 0 and tiaozhenCountFinal < 10:
								labelfile_name = video_name.replace('.mp4', '_') + "00000" + str(tiaozhenCountFinal) + ".txt"
							elif tiaozhenCountFinal >= 10 and tiaozhenCountFinal < 100:
								labelfile_name = video_name.replace('.mp4', '_') + "0000" + str(tiaozhenCountFinal) + ".txt"
							elif tiaozhenCountFinal >= 100 and tiaozhenCountFinal < 1000:
								labelfile_name = video_name.replace('.mp4', '_') + "000" + str(tiaozhenCountFinal) + ".txt"
							elif tiaozhenCountFinal >= 1000 and tiaozhenCountFinal < 10000:
								labelfile_name = video_name.replace('.mp4', '_') + "00" + str(tiaozhenCountFinal) + ".txt"
							elif tiaozhenCountFinal >= 10000 and tiaozhenCountFinal < 100000:
								labelfile_name = video_name.replace('.mp4', '_') + "0" + str(tiaozhenCountFinal) + ".txt"
							elif tiaozhenCountFinal >= 100000 and tiaozhenCountFinal < 1000000:
								labelfile_name = video_name.replace('.mp4', '_') + str(tiaozhenCountFinal) + ".txt"

							fp = open(os.path.join(output_video_path+"/image_platetxt",labelfile_name), "a")
							center_x = ((p_tl_x + p_br_x)/2)/float(img_w)
							center_y = ((p_tl_y + p_br_y)/2)/float(img_h)
							plate_w = (p_br_x - p_tl_x)/float(img_w)
							plate_h = (p_br_y - p_tl_y)/float(img_h)
							conf_score = frameDir[tiaozhenCountFinal][car]["platelocal"][8]
							labelInfo = [1,conf_score,center_x,center_y,plate_w,plate_h]
							fp.write(' '.join(map(str, labelInfo)))
							fp.write("\n")
							fp.close()

				# cv2.putText(arr, str(car), (int(frameDir[tiaozhenCountFinal][car]["platelocal"][4]), int(frameDir[tiaozhenCountFinal][car]["platelocal"][5])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
				cv2.putText(arr, str(tiaozhenCountFinal), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
			vid_writer.write(arr)
		tiaozhenCountFinal = tiaozhenCountFinal + 1
		if tiaozhenCountFinal > len(frameDir) * tiaozhen:
			break
	print("get labeltxt end !!!")
	vid.release()



if __name__ == '__main__':

	try:

		input_dir = sys.argv[1]
		# input_dir  = "samples/test"
		output_dir = sys.argv[2]
		# output_dir = "tmp/output"
		# car_track = "car_track_res.dat"

		lp_threshold = .5
		car_area_threshold = 3250
		plate_area_threshold = 1800
		tiaozhen = 1

		wpod_net_path = "model/lp-detector/wpod-net_update1.h5"
		wpod_net = load_model(wpod_net_path)

		trainList, trainLabel = loadDataSet("data/training_data.txt")
		bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=10)
		bdt.fit(trainList, trainLabel)

		if not isdir(output_dir):
			makedirs(output_dir)

		print
		'Searching for vehicles using YOLO...'

		file_name_list = [file_name for file_name in os.listdir(input_dir) \
						  if file_name.lower().endswith('mp4')]
		for file_name in file_name_list:

			vid = cv2.VideoCapture(os.path.join(input_dir, file_name))
			# out_file = os.path.join(output_dir, file_name.replace('.mp4', '_xiangdui_result.avi'))
			# out_file_middle = os.path.join(output_dir, file_name.replace('.mp4', '_middle_result.avi'))
			# fps = vid.get(cv2.CAP_PROP_FPS)  
			# vid_writer = cv2.VideoWriter(out_file_middle, \
			# 							 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(fps/1), \
			# 							 (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), \
			# 							  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

			tiaozhenCount = 1
			timeF = tiaozhen  
			
			car_track = file_name.replace('.mp4', '.dat')
			carDir = getCarsDir(os.path.join(input_dir, car_track))
			print("plate detection start!!!")
			while True:
				return_value, arr = vid.read()
				if not return_value:
					break

				if (tiaozhenCount % timeF == 0) and tiaozhenCount in carDir.keys(): 
					img = nparray_to_image(arr)

					if len(carDir[tiaozhenCount]):
						print("frameId:",tiaozhenCount)

						Iorig = arr
						testimg = Iorig
						height, width, c = Iorig.shape
						WH = np.array(Iorig.shape[1::-1], dtype=float)
						Lcars = []
						Lplates = []

						for carId in carDir[tiaozhenCount]:
							car_w = carDir[tiaozhenCount][carId]["carlocal"][4] - carDir[tiaozhenCount][carId]["carlocal"][0]
							car_h = carDir[tiaozhenCount][carId]["carlocal"][5] - carDir[tiaozhenCount][carId]["carlocal"][1]
							if car_w * car_h > car_area_threshold:
								tl = np.array([float(carDir[tiaozhenCount][carId]["carlocal"][0]) / width,
											   float(carDir[tiaozhenCount][carId]["carlocal"][1]) / height])
								
								if carDir[tiaozhenCount][carId]["carlocal"][5]+20<height:
									br = np.array([float(carDir[tiaozhenCount][carId]["carlocal"][4]) / width,
													float(carDir[tiaozhenCount][carId]["carlocal"][5]+20) / height])
								else:
									br = np.array([float(carDir[tiaozhenCount][carId]["carlocal"][4]) / width,
												   float(height) / height])

								label = Label(0, tl, br)  
								# draw_label(testimg, label, color=(0, 255, 255), thickness=2)
								# cv2.putText(testimg, str(carId), (int(carDir[tiaozhenCount][carId]["carlocal"][0]),
								# 								  carDir[tiaozhenCount][carId]["carlocal"][1]),
								# 			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
								# cv2.imwrite('%s/getRes.png' % output_dir, testimg)

								Icar = crop_region(Iorig, label) 

								ratio = float(max(Icar.astype(np.uint8).shape[:2])) / min(Icar.astype(np.uint8).shape[:2])
								side = int(ratio * 288.)
								bound_dim = min(side + (side % (2 ** 4)), 608)

								Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Icar.astype(np.uint8)), bound_dim, 2 ** 4,
															(240, 80),
															lp_threshold)

								if len(LlpImgs):
									# Ilp = LlpImgs[0]
									# Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
									# Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
									s = Shape(Llp[0].pts)

									pts = s.pts * label.wh().reshape(2, 1) + label.tl().reshape(2, 1)
									ptspx = pts * np.array(testimg.shape[1::-1], dtype=float).reshape(2, 1)
									# cv2.rectangle(testimg, (int(ptspx[0][0]),int(ptspx[1][0])),
									# 			  (int(ptspx[0][2]),int(ptspx[1][2])), (0, 255, 0), 2)


									# fp = open(os.path.join(output_dir+"/video_platetxt", file_name.replace('.mp4', '.txt')), "a")
									# plateinfo = [tiaozhenCount, carId,ptspx[0][0],ptspx[1][0], ptspx[0][1],
									# 			  ptspx[1][1], ptspx[0][2],ptspx[1][2], ptspx[0][3],ptspx[1][3], Llp[0].prob()]  
									# fp.write(', '.join(map(str, plateinfo)))
									# fp.write("\n")
									# fp.close()

									tl_point = isInside(carDir[tiaozhenCount][carId]["carlocal"][0], carDir[tiaozhenCount][carId]["carlocal"][1],
														carDir[tiaozhenCount][carId]["carlocal"][4], carDir[tiaozhenCount][carId]["carlocal"][5]+20, ptspx[0][0], ptspx[1][0])
									tr_point = isInside(carDir[tiaozhenCount][carId]["carlocal"][0], carDir[tiaozhenCount][carId]["carlocal"][1],
														carDir[tiaozhenCount][carId]["carlocal"][4], carDir[tiaozhenCount][carId]["carlocal"][5]+20, ptspx[0][1], ptspx[1][1])
									br_point = isInside(carDir[tiaozhenCount][carId]["carlocal"][0], carDir[tiaozhenCount][carId]["carlocal"][1],
														carDir[tiaozhenCount][carId]["carlocal"][4], carDir[tiaozhenCount][carId]["carlocal"][5]+20, ptspx[0][2], ptspx[1][2])
									bl_point = isInside(carDir[tiaozhenCount][carId]["carlocal"][0], carDir[tiaozhenCount][carId]["carlocal"][1],
														carDir[tiaozhenCount][carId]["carlocal"][4], carDir[tiaozhenCount][carId]["carlocal"][5]+20, ptspx[0][3], ptspx[1][3])
									if tl_point and tr_point and br_point and bl_point:
										carDir[tiaozhenCount][carId]["platelocal"].extend([ptspx[0][0],ptspx[1][0]])
										carDir[tiaozhenCount][carId]["platelocal"].extend([ptspx[0][1],ptspx[1][1]])
										carDir[tiaozhenCount][carId]["platelocal"].extend([ptspx[0][2],ptspx[1][2]])
										carDir[tiaozhenCount][carId]["platelocal"].extend([ptspx[0][3],ptspx[1][3]])
										
										carDir[tiaozhenCount][carId]["platelocal"].extend([Llp[0].prob()])
										
										index_list = []
										index_list.extend(carDir[tiaozhenCount][carId]["carlocal"])
										index_list.extend(carDir[tiaozhenCount][carId]["platelocal"])
										index_list = get_test_sampling(index_list)
										plate_res = bdt.predict([index_list])
										
										carDir[tiaozhenCount][carId]["platelocal"].extend(plate_res)

					# vid_writer.write(testimg)
				tiaozhenCount = tiaozhenCount + 1

			vid.release()
			print("plate detection end!!!")
			drawplate(carDir,input_dir, file_name,output_dir)


	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
