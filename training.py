
sourceImgsList = '/home/sains/unsupervised_flow/source_names.txt'
targetImgsList = '/home/sains/unsupervised_flow/target_names.txt'

logDirectory     = '/home/sains/unsupervised_flow/log_directory'
checkpointPath   = '/home/sains/unsupervised_flow/log_directory/swaayatt/model-7'
outputPath_image = '/home/sains/unsupervised_flow/results'
outputPath_flow  = '/home/sains/unsupervised_flow/results1'
targetLanePath   = '/home/sains/unsupervised_flow/results2'

import tensorflow as tf 
import numpy as np 
import time
import cv2

from components2 import *

class hyperParameters:

	def __init__(self):

		# params required for obtain training data

		self.batchSize      = 2
		self.contrastMin    = 0.7
		self.contrastMax    = 1.3
		self.brightnessStd  = 0.2 
		self.colorMin       = 0.9
		self.colorMax       = 1.1
		self.gammaMin       = 0.7
		self.gammaMax       = 1.5
		self.noiseStd       = 0.00

		self.borderThicknessH = 0.1 
		self.borderThicknessW = 0.1 

		self.nRows = 256
		self.nCols = 512

		# params required for building the network

		self.resnet    = True
		self.flowScale = 20.0

		# params required for constructing the losses

		self.weightDecay    = 0.0

		self.photoParamsRobustness = 0.53
		self.photoParamsScale      = 360.0

		self.smooth2ndParamsRobustness = 0.21 
		self.smooth2ndParamsScale      = 1.0
		self.smooth2ndParamsWeight     = 0.53

		self.smoothOccParamsRobustness = 0.9
		self.smoothOccParamsScale      = 1.8

		self.smoothParamsRobustness = 0.28
		self.smoothParamsScale      = 3.5
		self.smoothParamsWeight     = 0.64

		self.gradParamsWeight     = 6.408
		self.gradParamsRobustness = 0.46
		self.gradParamsScale      = 255

		self.boundaryAlpha  = 6.5

		self.lossComponentsBoundaries       = True
		self.lossComponentsGradient         = True
		self.lossComponentsSmooth2nd        = True
		self.lossComponentsBackward         = True
		self.lossComponentsAsymmetricSmooth = True

		# params required for training

		self.baseLearningRate = 1e-05
		self.epochs           = 250
		self.momentum1        = 0.9
		self.momentum2        = 0.999

def train(params, SourceList, TargetList, checkpointPath, logDirectory, outputPath_image, outputPath_flow):

	with open(SourceList) as f:

		source_data_list_path = [x[:-1] for x in f.readlines()]

	with open(TargetList) as f:

		target_data_list_path = [x[:-1] for x in f.readlines()]

	numSamples = len(source_data_list_path)

	numIterationsPerEpoch = int(numSamples / params.batchSize)

	numIterations = numIterationsPerEpoch * params.epochs 

	batchIds = np.arange(numSamples)

	iteration = 0

	sess = sessionSetup()

	sourceImages   = tf.placeholder(shape = [params.batchSize, params.nRows, params.nCols, 3], dtype = tf.float32, name = "source_images_list")
	targetImages   = tf.placeholder(shape = [params.batchSize, params.nRows, params.nCols, 3], dtype = tf.float32, name = "target_images_list")
	
	learningRate   = tf.placeholder(shape = [], dtype = tf.float32, name = "learning_rate" )
	recLossBWeight = tf.placeholder(shape = [], dtype = tf.float32, name = "recLossBWeight")

	trainData = trainingData(params, sourceImages, targetImages)

	with tf.variable_scope("netShare"):

		networkBodyF = NetworkBody(trainData, params)

	with tf.variable_scope("netShare", reuse = True):

		networkBodyB = NetworkBody(trainData, params, flipInput = True)

	trainingLoss = TrainingLoss(params, networkBodyF, networkBodyB, trainData, recLossBWeight)

	solver = attachSolver(trainingLoss.loss, params, learningRate)

	init_op = tf.global_variables_initializer()
	
	sess.run(init_op)

	train_saver = tf.train.Saver()

	if(checkpointPath != ''):

		print('loading pretrained weights')
		train_saver.restore(sess, checkpointPath)


	for e in range(params.epochs):

		shuffledIds = np.random.shuffle(batchIds)

		shuffledIds = batchIds

		totalEpe  = 0
		addedLoss = 0 
		totalSSIM = 0
		totalEpe2 = 0

		time.sleep(10)
		print(numIterationsPerEpoch)

		for batch in range(numIterationsPerEpoch):

			batchDataList = shuffledIds[params.batchSize * batch : params.batchSize * (batch + 1)]

			sourceImgList = [source_data_list_path[batchDataList[i]] for i in range(params.batchSize)]
			targetImgList = [target_data_list_path[batchDataList[i]] for i in range(params.batchSize)]

			#print(sourceImgList)
			#print(targetImgList)

			lr = learningRateSchedule(params.baseLearningRate, iteration) 

			rbw = unsupLossBSchedule(iteration)

			sourceImgs = [cv2.resize(cv2.imread(sourceImgList[i]), (params.nCols, params.nRows)) for i in range(params.batchSize)]
			targetImgs = [cv2.resize(cv2.imread(targetImgList[i]), (params.nCols, params.nRows)) for i in range(params.batchSize)]
			
			result, totalLoss, epe, ssimLoss, image1, image2, flowF, flowB, epe2 = sess.run((solver, trainingLoss.loss, trainingLoss.epe, trainingLoss.ssimloss, trainingLoss.img0, trainingLoss.img1, trainingLoss.flowForward, trainingLoss.flowBackward, trainingLoss.epe2Cost), feed_dict = {sourceImages : sourceImgs,  targetImages : targetImgs, learningRate : lr, recLossBWeight : rbw})

			addedLoss += totalLoss

			totalEpe += epe

			totalSSIM += ssimLoss 
 
			totalEpe2 += epe2

			# displaying and storing results.

			if batch % 10 == 0 and batch > 0:

				avgEpe  = totalEpe  / (batch + 1)
				avgLoss = addedLoss / (batch + 1)
				avgSSIM = totalSSIM / (batch + 1)
				avgEpe2 = totalEpe2 / (batch + 1)

				content = str("epoch : " + str(e) + "  " + "batch : " + str(batch) + "  " + "iteration : " + str(iteration) + "  " + "loss : " + str.format('{0:.6f}', avgLoss) + "  " + "EPE : " + str.format('{0:.6f}', avgEpe[0]) + "  " + "SSIM : " + str.format('{0:.6f}', avgSSIM) + "  " + "EPE2 : " + str.format('{0:.6f}', avgEpe2))

				print(content)

			if batch % 100 == 0 and batch > 0:

				gt0  = image1[0, 0, :, :, :] 
				gen0 = image1[1, 0, :, :, :]
				gt1  = image2[0, 0, :, :, :]
				gen1 = image2[1, 0, :, :, :]

				output_image = np.vstack((np.hstack((gt0 * 255.0 * 3, gt1 * 255.0 * 3)), np.hstack((gen0 * 255.0 * 3, gen1 * 255.0 * 3))))

				imageName = outputPath_image + '/' + str(e) + '_' + str(batch) + '.jpg'

				cv2.imwrite(imageName, output_image)

				flow0 = flowF[0, :, :, :]
				flow1 = flowB[0, :, :, :]

				output_flow = np.hstack((flow0 * 255.0, flow1 * 255))

				flowName = outputPath_flow + '/' + str(e) + '_' + str(batch) + '.jpg'

				cv2.imwrite(flowName, output_flow)

			iteration += 1

		train_saver.save(sess, logDirectory + '/' + 'swaayatt' + '/model', global_step = e)

def generate_image(image, Xoffset, Yoffset):

	# function generates target/source image by taking source/target image as input and the corresponding XShifts and YShifts
	# uses bilinear sampling technique

	s         = tf.shape(image)
	batchsize = s[0]
	N_ROWS    = s[1]
	N_COLS    = s[2]
	NUM_INPUT_CHANNELS = 3
			
	hf = tf.cast(N_ROWS, dtype=tf.float32)
	wf = tf.cast(N_COLS, dtype=tf.float32)

	image_flat = tf.reshape(image,   [-1, NUM_INPUT_CHANNELS])
	x_offset   = tf.reshape(Xoffset, [-1, N_ROWS, N_COLS]    )
	y_offset   = tf.reshape(Yoffset, [-1, N_ROWS, N_COLS]    )

	xf, yf = tf.meshgrid(tf.cast(tf.range(N_COLS), tf.float32), tf.cast(tf.range(N_ROWS), tf.float32))
	xf     = tf.tile(xf, tf.stack([batchsize, 1]))
	yf     = tf.tile(yf, tf.stack([batchsize, 1]))
	xf     = tf.reshape(xf, [-1, N_ROWS, N_COLS])
	yf     = tf.reshape(yf, [-1, N_ROWS, N_COLS])
	xf     = xf + x_offset 
	yf     = yf + y_offset 
			
	a = tf.expand_dims(xf - tf.floor(xf), axis=-1)
	b = tf.expand_dims(yf - tf.floor(yf), axis=-1)

	a = a * tf.zeros_like(a)
	b = b * tf.zeros_like(b)

	xl = tf.clip_by_value(tf.cast(tf.floor(xf), dtype=tf.int32), 0, N_COLS - 1)
	yt = tf.clip_by_value(tf.cast(tf.floor(yf), dtype=tf.int32), 0, N_ROWS - 1)
	xr = tf.clip_by_value(xl + 1, 0, N_COLS - 1)
	yb = tf.clip_by_value(yt + 1, 0, N_ROWS - 1)

	batch_ids = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(batchsize), axis=-1), axis=-1), [1, N_ROWS, N_COLS])

	idx_lt    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yt * N_COLS + xl, [-1]), tf.int32)
	idx_lb    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yb * N_COLS + xl, [-1]), tf.int32)
	idx_rt    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yt * N_COLS + xr, [-1]), tf.int32)
	idx_rb    = tf.cast(tf.reshape(batch_ids * N_ROWS * N_COLS + yb * N_COLS + xr, [-1]), tf.int32)

	val = tf.zeros_like(a)   
	val += (1 - a) * (1 - b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_lt), tf.float32), [-1, N_ROWS, N_COLS, NUM_INPUT_CHANNELS]) 
	val += (1 - a) * (0 + b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_lb), tf.float32), [-1, N_ROWS, N_COLS, NUM_INPUT_CHANNELS])
	val += (0 + a) * (1 - b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_rt), tf.float32), [-1, N_ROWS, N_COLS, NUM_INPUT_CHANNELS])
	val += (0 + a) * (0 + b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_rb), tf.float32), [-1, N_ROWS, N_COLS, NUM_INPUT_CHANNELS])

	return val 


def test(params, SourceLoc, TargetLoc, checkpointPath):

	sess = sessionSetup()

	params.batchSize = 1

	sourceImages   = tf.placeholder(shape = [params.batchSize, params.nRows, params.nCols, 3], dtype = tf.float32, name = "source_images_list")
	targetImages   = tf.placeholder(shape = [params.batchSize, params.nRows, params.nCols, 3], dtype = tf.float32, name = "target_images_list")

	trainData = trainingData(params, sourceImages, targetImages)

	with tf.variable_scope("netShare"):

		networkBodyF = NetworkBody(trainData, params)

	with tf.variable_scope("netShare", reuse = True):

		networkBodyB = NetworkBody(trainData, params, flipInput = True)

	init_op = tf.global_variables_initializer()
	
	sess.run(init_op)

	train_saver = tf.train.Saver()

	train_saver.restore(sess, checkpointPath)

	sourceImgs = [cv2.resize(cv2.imread(SourceLoc), (params.nCols, params.nRows)) for i in range(params.batchSize)]
	targetImgs = [cv2.resize(cv2.imread(TargetLoc), (params.nCols, params.nRows)) for i in range(params.batchSize)]
			
	TSFlow, STFlow = sess.run((networkBodyF.flows, networkBodyB.flows), feed_dict = {sourceImages : sourceImgs,  targetImages : targetImgs})

	#sess.close()
	tf.reset_default_graph()

	return TSFlow, STFlow



def main():
	
	params = hyperParameters()

	train(params, sourceImgsList, targetImgsList, checkpointPath, logDirectory, outputPath_image, outputPath_flow)

main()