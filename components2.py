"""
Jai Shri Ram
Author - Sai Shankar Narasimhan
Date   - 12th May, 2019

The program contains all the building blocks (functions and classes) of the unsupflow network

Paper Name (For future reference) - " Back to Basics: Unsupervised Learning of Optical Flow via BrightnessConstancy and Motion Smoothness "

Functions
1.  fullyConnectedLayer
1.  fullyConnectedLayerRelu
1.  convLayer
2.  leakyRelu
3.  convLayerRelu
4.  deconvLayer
5.  deconvLayerRelu
6.  batchMeshGrid
7.  batchMeshGridLike
8.  photoAug
9.  photoAugParam
10. charbonnierLoss
11. epeEval
11. SSIM
12. borderOcclusionMask
13. validPixelMask
14. flowTransformGrid
15. flowWarp
16. flowRefinementConcat
17. flowToRgb
18. rgbToGray
19. gradientFromGray
20. resSkip
21. resLayer
22. resLayerStride
23. resBlock
24. learningRateSchedule
25. sessionSetup
26. solver
27. unsuppLossBSchedule
28. smoothLossMaskCorrection
29. smoothLoss
30. smoothLoss2ndMaskCorrection
31. smoothLoss2nd
32. photoLoss
33.	gradLoss
34. asymmetricSmoothLoss
35. unsupFLowLoss

Classes

1. TrainingData
	- constructor

1. NetworkBody
	- constructor
	- buildNetwork

2. TrainingLoss

"""

import tensorflow as tf 
import math

def fullyConnectedLayer(x, outUnits):

	with tf.variable_scope(None, default_name = "fc"):

		inUnits = x.get_shape()[1]

		fcShape = [inUnits, outUnits]

		w = tf.get_variable("fcWeights", shape = fcShape,     initializer = tf.uniform_unit_scaling_initializer())
		b = tf.get_variable("fcBiases",  shape = [outUnits],   initializer = tf.initializers.ones())

		tf.add_to_collection("fcWeights", w)
		tf.add_to_collection("fcBiases", b)

		fc = tf.matmul(x, w) 

		return fc + b

def fullyConnectedLayerRelu(x, outUnits):

	with tf.variable_scope(None, default_name = "reluFc"):

		return leakyRelu(fullyConnectedLayer(x, outUnits), 0.1)

def convLayer(x, kernelSize, outMaps, stride):

	with tf.variable_scope(None, default_name = "conv"):

		inMaps = x.get_shape()[3]
		
		kernelShape = [kernelSize, kernelSize, inMaps, outMaps]

		w = tf.get_variable("weights", shape = kernelShape, initializer = tf.uniform_unit_scaling_initializer())
		b = tf.get_variable("biases",  shape = [outMaps],   initializer = tf.constant_initializer(0))

		tf.add_to_collection("weights", w)
		tf.add_to_collection("biases", b)

		conv = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding = "SAME", name = "conv2d")

		return conv + b

def leakyRelu(x, alpha):

	with tf.variable_scope(None, default_name = "leakyRelu"):

		return tf.maximum(x, x * alpha)

def convLayerRelu(x, kernelSize, outMaps, stride):

	with tf.variable_scope(None, default_name = "reluConv"):

		return leakyRelu(convLayer(x, kernelSize, outMaps, stride), 0.1)

def deconvLayer(x, kernelSize, outMaps, stride):

	with tf.variable_scope(None, default_name = "deconv"):

		inMaps = x.get_shape()[3]

		kernelShape = [kernelSize, kernelSize, outMaps, inMaps]

		w = tf.get_variable("weights", shape = kernelShape, initializer = tf.uniform_unit_scaling_initializer())

		tf.add_to_collection("weights", w)

		inShape  = tf.shape(x)
		outShape = tf.stack([inShape[0], inShape[1] * stride, inShape[2] * stride, outMaps], name = "shapeEval")

		deconv = tf.nn.conv2d_transpose(x, w, outShape, [1, stride, stride, 1], padding = "SAME", name = "deconv")

		return deconv

def deconvLayerRelu(x, kernelSize, outMaps, stride):

	with tf.variable_scope(None, default_name = "reluDeconv"):

		return leakyRelu(deconvLayer(x, kernelSize, outMaps, stride), 0.1)


def batchMeshGrid(batchSize, height, width):

	with tf.variable_scope(None, default_name = "batchMeshGrid"):

		X, Y = tf.meshgrid(tf.linspace(0.0, tf.cast(width, dtype = tf.float32) - 1.0, width), tf.linspace(0.0 , tf.cast(height, dtype = tf.float32) - 1.0 , height))		

		X = tf.cast(X, dtype = tf.float32)
		Y = tf.cast(Y, dtype = tf.float32)

		grid = tf.stack([X, Y], axis = -1)

		batchGrid = tf.expand_dims(grid, axis = 0)
		batchGrid = tf.tile(batchGrid, [batchSize, 1, 1, 1])

		return batchGrid

def batchMeshGridLike(tensor):

	with tf.variable_scope(None, default_name = "batchMeshGridLike"):

		shape = tensor.get_shape()

		return batchMeshGrid(shape[0].value, shape[1].value, shape[2].value)

def photoAug(image, augParams):

	with tf.variable_scope(None, default_name = "photoAug"):

		contrast   = augParams[0]
		brightness = augParams[1]
		color      = augParams[2]
		gamma      = augParams[3]
		noiseStd   = augParams[4]

		noise = tf.random_normal(image.get_shape(), 0, noiseStd)

		image = image * contrast + brightness
		image = image * color
		image = tf.maximum(tf.minimum(image, 1), 0) 
		image = tf.pow(image, 1 / gamma)
		image = image + noise

		return image

def photoAugParam(batchSize, contrastMin, contrastMax, brightnessStd, colorMin, colorMax, gammaMin, gammaMax, noiseStd):

	with tf.variable_scope(None, default_name = "photoAugParam"):

		contrast   = tf.random_uniform([batchSize, 1, 1, 1], contrastMin, contrastMax)
		brightness = tf.random_normal ([batchSize, 1, 1, 1], 0, brightnessStd)
		color      = tf.random_uniform([batchSize, 1, 1, 3], colorMin, colorMax)
		gamma      = tf.random_uniform([batchSize, 1, 1, 1], gammaMin, gammaMax)

		noise = noiseStd

		return [contrast,brightness,color,gamma,noise]

def charbonnierLoss(x, alpha, beta, epsilon):

	with tf.variable_scope(None, default_name = "charbonnierLoss"):

		epsilonSq = epsilon * epsilon
		xScale    = x * beta

		return tf.pow(xScale * xScale + epsilonSq, alpha)

def epeEval(predicted, truth, mask):

	with tf.variable_scope(None, default_name = "epeEval"):

		difference      = predicted - truth
		differenceSq    = difference * difference 
		differenceSqSum = tf.reduce_sum(differenceSq, reduction_indices = [3], keep_dims = True)

		epe = tf.sqrt(differenceSqSum)

		maskedFlow = epe * mask

		validPixels = tf.reduce_sum(mask, reduction_indices = [0, 1, 2])

		epeSum = tf.reduce_sum(maskedFlow, reduction_indices = [0, 1, 2])

		avgEpe = epeSum / validPixels

		return avgEpe

def SSIM(target, source, occInvalidMask):

	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	source = source * occInvalidMask
	target = target * occInvalidMask

	mean_source = tf.layers.average_pooling2d(source, (3, 3), (1, 1), 'VALID')
	mean_target = tf.layers.average_pooling2d(target, (3, 3), (1, 1), 'VALID')

	sigma_source  = tf.layers.average_pooling2d(source ** 2,      (3, 3), (1, 1), 'VALID') - mean_source ** 2
	sigma_target  = tf.layers.average_pooling2d(target ** 2,      (3, 3), (1, 1), 'VALID') - mean_target ** 2
	sigma_st      = tf.layers.average_pooling2d(source * target , (3, 3), (1, 1), 'VALID') - mean_source * mean_target

	SSIM_n = (2 * mean_source * mean_target + C1) * (2 * sigma_st + C2)
	SSIM_d = (mean_source ** 2 + mean_target ** 2 + C1) * (sigma_source + sigma_target + C2)

	SSIM = SSIM_n / SSIM_d

	return tf.reduce_mean(SSIM), tf.reduce_sum(tf.clip_by_value((1 - SSIM) / 2, 0, 1))

def borderOcclusionMask(flow):

	with tf.variable_scope(None, default_name = "matchableFlowMask"):

		flowShape = flow.get_shape()

		x = flowShape[2].value
		y = flowShape[1].value

		X, Y = tf.meshgrid(tf.linspace(0.0, tf.cast(x, dtype = tf.float32) - 1.0, x), tf.linspace(0.0 , tf.cast(y, dtype = tf.float32) - 1.0 , y))

		X = tf.cast(X, dtype = tf.float32)
		Y = tf.cast(Y, dtype = tf.float32)

		grid = tf.stack([X, Y], axis = -1)
		grid = tf.expand_dims(grid, axis = 0)
		grid = tf.tile(grid, [flowShape[0].value, 1, 1, 1])

		flowPoints  = grid + flow
		flowPointsU = tf.expand_dims(flowPoints[:, :, :, 0], axis = -1)
		flowPointsV = tf.expand_dims(flowPoints[:, :, :, 1], axis = -1)

		mask1 = tf.greater(flowPointsU, 0)
		mask2 = tf.greater(flowPointsV, 0)
		mask3 = tf.less(flowPointsU, flowShape[2].value - 1)
		mask4 = tf.less(flowPointsV, flowShape[1].value - 1)

		mask = tf.logical_and(mask1, mask2)
		mask = tf.logical_and(mask , mask3)
		mask = tf.logical_and(mask , mask4)

		mask = tf.cast(mask, dtype = tf.float32)

		return mask

def validPixelMask(lossShape, borderPercentH, borderPercentW):

	with tf.variable_scope(None, default_name = "validPixelMask"):

		batchSize = tf.cast(lossShape[0], dtype = tf.int32) 
		height    = tf.cast(lossShape[1], dtype = tf.int32)
		width     = tf.cast(lossShape[2], dtype = tf.int32)
		channels  = tf.cast(lossShape[3], dtype = tf.int32)

		borderThicknessH = tf.cast(tf.round(borderPercentH * tf.cast(height, dtype = tf.float32)), dtype = tf.int32)
		borderThicknessW = tf.cast(tf.round(borderPercentW * tf.cast(width , dtype = tf.float32)), dtype = tf.int32)

		innerHeight = height - 2 * borderThicknessH
		innerWidth  = width  - 2 * borderThicknessW

		topBottom = tf.zeros(tf.stack([batchSize, borderThicknessH, innerWidth, channels]))
		leftRight = tf.zeros(tf.stack([batchSize, height, borderThicknessW, channels]))
		centre    = tf.ones (tf.stack([batchSize, innerHeight, innerWidth, channels]))

		mask = tf.concat([topBottom, centre, topBottom], 1)
		mask = tf.concat([leftRight, mask, leftRight], 2)

		ref = tf.zeros(lossShape)
		mask.set_shape(ref.get_shape())

		return mask

def flowTransformGrid(flow):

	with tf.variable_scope(None, default_name = "flowTransformGrid"):

		resampleGrid = batchMeshGridLike(flow) + flow 

		return resampleGrid

def flowWarp(data, flow):

	with tf.variable_scope(None, default_name = "flowWarp"):

		resampleGrid = flowTransformGrid(flow)

		warped = tf.contrib.resampler.resampler(data,resampleGrid)

		return warped

def flowRefinementConcat(prev, skip, flow):

	with tf.variable_scope(None, default_name = "flowRefinementConcat"):

		with tf.variable_scope(None, default_name = "upsampleFlow"):

			upsample = deconvLayer(flow, 4, 2, 2)

		return tf.concat([prev, skip, upsample], 3)

def flowToRgb(flow, zeroFlow = "value"):

	with tf.variable_scope(None, default_name = "flowToRgb"):

		mag    = tf.sqrt(tf.reduce_sum(flow ** 2, axis = -1))
		ang180 = tf.atan2(flow[:, :, :, 1], flow[:, :, :, 0])
		ones   = tf.ones_like(mag)

		ang = ang180 * tf.cast(tf.greater_equal(ang180, 0) , dtype = tf.float32)		
		ang = ang + (ang180 + 2 * math.pi) * tf.cast(tf.less(ang180, 0), tf.float32)

		largestMag = tf.expand_dims(tf.expand_dims(tf.reduce_max(mag, axis = [1, 2]), axis = -1), axis = -1)

		magNorm = mag / largestMag
		angNorm = ang / (math.pi*2)

		if zeroFlow == "value":
				
				hsv = tf.stack([angNorm, ones, magNorm], axis = -1)
		
		elif zeroFlow == "saturation":
				
				hsv = tf.stack([angNorm, magNorm, ones], axis = -1)
		
		else:
				
				assert("zeroFlow mode must be {'value','saturation'}")
		
		rgb = tf.image.hsv_to_rgb(hsv)
		
		return rgb

def rgbToGray(img):

	with tf.variable_scope(None, default_name = "rgbToGray"):

		rgbWeights = [0.2989, 0.5870, 0.1140]

		rgbWeights = tf.expand_dims(tf.expand_dims(tf.expand_dims(rgbWeights, 0), 0), 0)

		weightedImg = img * rgbWeights

		return tf.reduce_sum(weightedImg, reduction_indices = [3], keep_dims = True)

def gradientFromGray(img):

	with tf.variable_scope(None, default_name = "forwardDifferencesSingle"):

		kernel = tf.transpose(tf.constant([\
												[\
													[\
														[0, 0,  0], \
														[0, 1, -1], \
														[0, 0,  0]  \
													] \
												],\
												[\
													[\
														[0,  0, 0], \
														[0,  1, 0], \
														[0, -1, 0]  \
													] \
												]\
										  ] , dtype = tf.float32), perm = [3, 2, 1, 0]) # [1, 2, 3, 3] before transpose and [3, 3, 2, 1] after transpose

		return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], padding = "SAME")

def resSkip(x, outMaps, stride):

	with tf.variable_scope(None, default_name = "resSkip"):

		return convLayer(x, 1, outMaps, stride) 

def resLayer(x, kernelSize, outMaps):

	with tf.variable_scope(None, default_name = "resLayer"):

		skip = x

		conv1 = convLayerRelu(x, kernelSize, outMaps, 1)
		conv2 = convLayer(conv1, kernelSize, outMaps, 1)

		return leakyRelu(conv2 + skip, 0.1)

def resLayerStride(x, kernelSize, outMaps):

	with tf.variable_scope(None, default_name = "resLayerStride"):

		skip = resSkip(x, outMaps, 2)

		conv1 = convLayerRelu(x, kernelSize, outMaps, 2)
		conv2 = convLayer(conv1, kernelSize, outMaps, 1)

		return leakyRelu(conv2 + skip, 0.1)

def resBlock(x, kernelSize, outMaps):

	with tf.variable_scope(None, default_name = "resBlock"):

		res1 = resLayerStride(x, kernelSize, outMaps)
		res2 = resLayer(res1, kernelSize, outMaps)
		
		return res2

def learningRateSchedule(baseLr, iteration):

	if iteration > 500000:

		return baseLr / 8

	elif iteration > 400000:

		return baseLr / 4

	elif iteration > 300000:

		return baseLr / 2

	elif iteration > 200000:

		return baseLr

	elif iteration > 100000:

		return baseLr

	else:

		return baseLr

def sessionSetup():
	
	config = tf.ConfigProto()
	
	config.gpu_options.allow_growth = True
	
	#config.allow_soft_placement = True
	
	#config.log_device_placement = True

	return tf.Session(config=config)

def attachSolver(loss, params, learningRate):

	with tf.variable_scope(None, default_name = "solver"):

		#learningRate = tf.placeholder(dtype = tf.float32, shape = [])
		
		solver = tf.train.AdamOptimizer(learning_rate = learningRate, beta1 = params.momentum1, beta2 = params.momentum2).minimize(loss)

		return solver

def unsupLossBSchedule(iteration):

	if iteration > 400000:
		
		return 0.5
	
	elif iteration > 300000:
		
		return 0.5
	
	elif iteration > 200000:
		
		return 0.5
	
	elif iteration > 190000:
		
		return 0.5
	
	elif iteration > 180000:
		
		return 0.4
	
	elif iteration > 170000:
		
		return 0.3
	
	elif iteration > 160000:
		
		return 0.2
	
	elif iteration > 150000:
		
		return 0.1
	
	else:
		
		return 0.0

def smoothLossMaskCorrection(validMask):

	inclusionKernel = tf.transpose(tf.constant([\
													[\
														[\
															[0, 0, 0],\
															[0, 1, 1],\
															[0, 1, 0] \
														],\
													],\
												] , dtype = tf.float32), perm = [3, 2, 1, 0] )

	with tf.variable_scope(None, default_name = "smoothLossMaskCorrection"):

		maskCor = tf.nn.conv2d(validMask, inclusionKernel, [1, 1, 1, 1], padding = "SAME")
		maskCor = tf.greater_equal(maskCor, 2.95)
		maskCor = tf.cast(maskCor, dtype = tf.float32)

		return maskCor

def smoothLoss(flow, alpha, beta, validPixelMask = None, img0Grad = None, boundaryAlpha = 0):

	kernel = tf.transpose(tf.constant([\
											[\
													[\
														[0, 0,  0], \
														[0, 1, -1], \
														[0, 0,  0]  \
													] \
											],\
											[\
													[\
														[0,  0, 0], \
														[0,  1, 0], \
														[0, -1, 0]  \
													] \
											]\
									   ] , dtype = tf.float32), perm = [3, 2, 1, 0]) # [1, 2, 3, 3] before transpose and [3, 3, 2, 1] after transpose

	with tf.variable_scope(None, default_name = "smoothLoss"):

		u = tf.slice(flow, [0, 0, 0, 0], [-1, -1, -1,  1])
		v = tf.slice(flow, [0, 0, 0, 1], [-1, -1, -1, -1])

		flowShape = flow.get_shape()

		neighbourDiffU = tf.nn.conv2d(u, kernel, [1, 1, 1, 1], padding = "SAME")
		neighbourDiffV = tf.nn.conv2d(v, kernel, [1, 1, 1, 1], padding = "SAME")

		diffs = tf.concat([neighbourDiffV, neighbourDiffV], axis = 3)

		dists = tf.reduce_sum(tf.abs(diffs), axis = 3, keep_dims = True)

		robustLoss = charbonnierLoss(dists, alpha, beta, 0.001)

		if not img0Grad == None:
			
			dMag = tf.sqrt(tf.reduce_sum(img0Grad ** 2, axis = 3, keep_dims = True))
			mask = tf.exp(- boundaryAlpha * dMag)
			
			robustLoss *= mask

		if(validPixelMask == None):

			return robustLoss

		else:

			return robustLoss * smoothLossMaskCorrection(validPixelMask)

def smoothLoss2ndMaskCorrection(validMask):

	inclusionKernel = tf.transpose(tf.constant([\
													[\
														[\
															[0, 0, 0, 0, 0],\
															[0, 0, 0, 0, 0],\
															[0, 0, 1, 1, 1],\
															[0, 0, 1, 0, 0],\
															[0, 0, 1, 0, 0] \
														]\
													]\
												], dtype = tf.float32), perm = [3, 2, 1, 0])

	maskCor = tf.nn.conv2d(validMask, inclusionKernel, [1, 1, 1, 1], padding = "SAME")
	maskCor = tf.greater_equal(maskCor, 4.95)
	maskCor = tf.cast(maskCor, dtype = tf.float32)

	return maskCor 

def smoothLoss2nd(flow, alpha, beta, validPixelMask = None, img0Grad = None, boundaryAlpha = 0):

	kernel = tf.transpose(tf.constant([\
											[\
													[\
														[0, 0,  0], \
														[0, 1, -1], \
														[0, 0,  0]  \
													] \
											],\
											[\
													[\
														[0,  0, 0], \
														[0,  1, 0], \
														[0, -1, 0]  \
													] \
											]\
									   ] , dtype = tf.float32), perm = [3, 2, 1, 0]) # [1, 2, 3, 3] before transpose and [3, 3, 2, 1] after transpose

	with tf.variable_scope(None, default_name = "smoothLoss2nd"):

		u = tf.slice(flow, [0, 0, 0, 0], [-1, -1, -1,  1])
		v = tf.slice(flow, [0, 0, 0, 1], [-1, -1, -1, -1])

		flowShape = flow.get_shape()

		neighbourDiffU = tf.nn.conv2d(u, kernel, [1, 1, 1, 1], padding = "SAME")
		neighbourDiffV = tf.nn.conv2d(v, kernel, [1, 1, 1, 1], padding = "SAME")

		neighbourDiffU_x = tf.nn.conv2d(tf.expand_dims(neighbourDiffU[:, :, :, 0], axis = -1), kernel, [1, 1, 1, 1], padding = "SAME")
		neighbourDiffU_y = tf.nn.conv2d(tf.expand_dims(neighbourDiffU[:, :, :, 1], axis = -1), kernel, [1, 1, 1, 1], padding = "SAME")
		neighbourDiffV_x = tf.nn.conv2d(tf.expand_dims(neighbourDiffV[:, :, :, 0], axis = -1), kernel, [1, 1, 1, 1], padding = "SAME")
		neighbourDiffV_y = tf.nn.conv2d(tf.expand_dims(neighbourDiffV[:, :, :, 1], axis = -1), kernel, [1, 1, 1, 1], padding = "SAME")

		diffs = tf.concat([neighbourDiffU_x, neighbourDiffU_y, neighbourDiffV_x, neighbourDiffV_y], axis = 3)
		dists = tf.reduce_sum(tf.abs(diffs), axis = 3, keep_dims = True)

		robustLoss = charbonnierLoss(dists, alpha, beta, 0.001)

		if not img0Grad == None:
			
			dMag = tf.sqrt(tf.reduce_sum(img0Grad ** 2, axis = 3, keep_dims = True))
			mask = tf.exp(- boundaryAlpha * dMag)
			
			robustLoss *= mask

		if(validPixelMask == None):

			return robustLoss

		else:

			return robustLoss * smoothLoss2ndMaskCorrection(validPixelMask)

def photoLoss(flow, downsampledFrame0, downsampledFrame1, alpha, beta):

	with tf.variable_scope(None, default_name = "photoLoss"):

		flowShape = flow.get_shape()
		batchSize = flowShape[0]
		height    = flowShape[1]
		width     = flowShape[2]
		
		outshape  = tf.stack([height,width])

		warpedFrame2 = flowWarp(downsampledFrame1,flow)

		photoDiff = downsampledFrame0 - warpedFrame2

		photoDist = tf.reduce_sum(tf.abs(photoDiff), axis = 3, keep_dims = True)
		
		robustLoss = charbonnierLoss(photoDist, alpha, beta, 0.001)

		return robustLoss 

def gradLoss(flow, downsampledGrad0, downsampledGrad1, alpha, beta):

	with tf.variable_scope(None, default_name = "gradLoss"):

		return photoLoss(flow, downsampledGrad0, downsampledGrad1, alpha, beta)

def asymmetricSmoothLoss(flow, params, occMask, validPixelMask, img0Grad = None, boundaryAlpha = 0):

	with tf.variable_scope(None, default_name = "asymmetricSmoothLoss"):

		alpha = params.smoothParamsRobustness
		beta  = params.smoothParamsScale
		
		occAlpha = params.smoothOccParamsRobustness
		occBeta  = params.smoothOccParamsScale  

		flowValid   = flow * occMask
		flowInvalid = flow * (1.0 - occMask)

		flowValid  = tf.stop_gradient(flowValid)
		routedFlow = flowValid + flowInvalid

		occSmooth = smoothLoss(routedFlow, occAlpha, occBeta, None, img0Grad, boundaryAlpha)

		nonOccSmooth = smoothLoss(flow, alpha, beta, occMask, img0Grad, boundaryAlpha)

		valid = smoothLossMaskCorrection(validPixelMask)
		
		smooth = nonOccSmooth + occSmooth
		
		return smooth * valid

def unsupFlowLoss(flow, frame0, frame1, validPixelMask, params):

	with tf.variable_scope(None, default_name = "unsupFlowLoss"):

		photoAlpha = params.photoParamsRobustness
		photoBeta  = params.photoParamsScale

		smoothReg = params.smoothParamsWeight

		smooth2ndReg   = params.smooth2ndParamsWeight
		smooth2ndAlpha = params.smooth2ndParamsRobustness
		smooth2ndBeta  = params.smooth2ndParamsScale

		gradReg   = params.gradParamsWeight
		gradAlpha = params.gradParamsRobustness
		gradBeta  = params.gradParamsScale

		boundaryAlpha  = params.boundaryAlpha
		
		lossComponentsBoundaries       = params.lossComponentsBoundaries
		lossComponentsGradient         = params.lossComponentsGradient
		lossComponentsSmooth2nd        = params.lossComponentsSmooth2nd
		lossComponentsBackward         = params.lossComponentsBackward
		lossComponentsAsymmetricSmooth = params.lossComponentsAsymmetricSmooth

		rgb0 = frame0["rgbNorm"]
		rgb1 = frame1["rgbNorm"]
		
		img0 = frame0["rgb"]
		img1 = frame1["rgb"]

		grad0 = frame0["grad"]
		grad1 = frame1["grad"]

		occMask = borderOcclusionMask(flow) 

		occInvalidMask = validPixelMask * occMask 

		photo = photoLoss(flow, rgb0, rgb1, photoAlpha, photoBeta)
		
		grad = gradLoss(flow, grad0, grad1, gradAlpha, gradBeta)

		warpedFrame2 = flowWarp(rgb1, flow)

		genImg0 = flowWarp(img1, flow)

		imgGrad = None
		
		if lossComponentsBoundaries:

			imgGrad = grad0

		if lossComponentsAsymmetricSmooth:

			smoothMasked = asymmetricSmoothLoss(flow, params, occMask, validPixelMask, imgGrad, boundaryAlpha)
		
		else:

			smoothMasked = smoothLoss(flow, smoothAlpha, smoothBeta, validPixelMask, imgGrad, boundaryAlpha)

		smooth2ndMasked = smoothLoss2nd(flow, smooth2ndAlpha, smooth2ndBeta, validPixelMask, imgGrad, boundaryAlpha)

		endPointError = epeEval(warpedFrame2, rgb0, occInvalidMask)

		ssimError, ssimCost = SSIM(warpedFrame2, rgb0, occInvalidMask)

		photoMasked = photo * occInvalidMask
		
		gradMasked = grad * occInvalidMask

		out_img = tf.concat([tf.expand_dims(img0 * occInvalidMask, axis = 0), tf.expand_dims(genImg0 * occInvalidMask, axis = 0)], axis = 0)

		photoAvg     = tf.reduce_mean(photoMasked)
		gradAvg      = tf.reduce_mean(gradMasked)
		smoothAvg    = tf.reduce_mean(smoothMasked)
		smooth2ndAvg = tf.reduce_mean(smooth2ndMasked)

		gradAvg      = gradAvg * gradReg
		smoothAvg    = smoothAvg * smoothReg
		smooth2ndAvg = smooth2ndAvg * smooth2ndReg

		finalLoss = photoAvg + smoothAvg + (1 - ssimError) * 100

		epe2 = photoAvg + smoothAvg

		if lossComponentsSmooth2nd:

			finalLoss += smooth2ndAvg
		
		if lossComponentsGradient:

			finalLoss += gradAvg

		return finalLoss, endPointError, ssimError, out_img, epe2, occInvalidMask 

def generate_shift(image, Xoffset, Yoffset, params):

	s         = tf.shape(image)
	nrows     = params.nRows
	ncols     = params.nCols 
	batchsize = params.batchSize

	hf = tf.cast(nrows, dtype=tf.float32)
	wf = tf.cast(ncols, dtype=tf.float32)

	image_flat = tf.reshape(image,   [-1])
	x_offset   = tf.reshape(Xoffset, [-1, nrows, ncols])
	y_offset   = tf.reshape(Yoffset, [-1, nrows, ncols])

	xf, yf = tf.meshgrid(tf.cast(tf.range(ncols), tf.float32), tf.cast(tf.range(nrows), tf.float32))
	xf     = tf.tile(xf, tf.stack([batchsize, 1]))
	yf     = tf.tile(yf, tf.stack([batchsize, 1]))
	xf     = tf.reshape(xf, [-1, nrows, ncols])
	yf     = tf.reshape(yf, [-1, nrows, ncols])
	xf     = xf + x_offset 
	yf     = yf + y_offset 
			
	a = xf - tf.floor(xf)
	b = yf - tf.floor(yf)

	xl = tf.clip_by_value(tf.cast(tf.floor(xf), dtype=tf.int32), 0, ncols - 1)
	yt = tf.clip_by_value(tf.cast(tf.floor(yf), dtype=tf.int32), 0, nrows - 1)
	xr = tf.clip_by_value(xl + 1, 0, ncols - 1)
	yb = tf.clip_by_value(yt + 1, 0, nrows - 1)

	batch_ids = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(batchsize), axis=-1), axis=-1), [1, nrows, ncols])
	idx_lt    = tf.cast(tf.reshape(batch_ids * nrows * ncols + yt * ncols + xl, [-1]), tf.int32)
	idx_lb    = tf.cast(tf.reshape(batch_ids * nrows * ncols + yb * ncols + xl, [-1]), tf.int32)
	idx_rt    = tf.cast(tf.reshape(batch_ids * nrows * ncols + yt * ncols + xr, [-1]), tf.int32)
	idx_rb    = tf.cast(tf.reshape(batch_ids * nrows * ncols + yb * ncols + xr, [-1]), tf.int32)

	val = tf.zeros_like(a)   
	val += (1 - a) * (1 - b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_lt), tf.float32), [-1, nrows, ncols]) 
	val += (1 - a) * (0 + b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_lb), tf.float32), [-1, nrows, ncols])
	val += (0 + a) * (1 - b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_rt), tf.float32), [-1, nrows, ncols])
	val += (0 + a) * (0 + b) * tf.reshape(tf.cast(tf.gather(image_flat, idx_rb), tf.float32), [-1, nrows, ncols])

	return val 


def consistency(flowF, flowB):

	genFlowF = generate_shift() 
	genFlowB = generate_shift()

	flowFLoss = tf.reduce_sum(genFlowF - flowF)
	flowBLoss = tf.reduce_sum(genFlowB - flowB)

	consistencyLoss = flowFLoss + flowBLoss

	return consistencyLoss

class trainingData:

	def __init__(self, params, sourceInput, targetInput):

		sourceInput = tf.image.resize_images(sourceInput, [params.nRows, params.nCols], tf.image.ResizeMethod.AREA)
		targetInput = tf.image.resize_images(targetInput, [params.nRows, params.nCols], tf.image.ResizeMethod.AREA)

		augParams = photoAugParam(params.batchSize, params.contrastMin, params.contrastMax, params.brightnessStd, params.colorMin, params.colorMax, params.gammaMin, params.gammaMax, params.noiseStd)

		mean = [[[[0.448553, 0.431021, 0.410602]]]]
		
		img0raw = tf.cast(sourceInput, tf.float32) / 255.0 - mean
		img1raw = tf.cast(targetInput, tf.float32) / 255.0 - mean

		imData0aug = photoAug(img0raw, augParams) - mean
		imData1aug = photoAug(img1raw, augParams) - mean

		borderMask = validPixelMask(tf.stack([1, img0raw.get_shape()[1], img0raw.get_shape()[2], 1]), params.borderThicknessH, params.borderThicknessW)

		self.imData0aug = imData0aug * borderMask
		self.imData1aug = imData1aug * borderMask

		self.lrn0 = tf.nn.local_response_normalization(img0raw, depth_radius = 2, alpha = (1.0 / 1.0), beta = 0.7, bias = 1)
		self.lrn1 = tf.nn.local_response_normalization(img1raw, depth_radius = 2, alpha = (1.0 / 1.0), beta = 0.7, bias = 1)

		imData0Gray = rgbToGray(img0raw)
		imData1Gray = rgbToGray(img1raw)

		self.imData0Grad = gradientFromGray(imData0Gray)
		self.imData1Grad = gradientFromGray(imData1Gray)

		self.frame0 = {
				"rgb": self.imData0aug,
				"rgbNorm": self.lrn0,
				"grad": self.imData0Grad
			}

		self.frame1 = {
				"rgb": self.imData1aug,
				"rgbNorm": self.lrn1,
				"grad": self.imData1Grad
			}

		self.validMask = borderMask

class NetworkBody:
	
	def __init__(self, trainingData, params, flipInput = False):

		frame0 = trainingData.frame0["rgb"]
		frame1 = trainingData.frame1["rgb"]
		
		if flipInput:
			
			combined = tf.concat([frame1,frame0], 3)						#[N H W 6]
		
		else:
			
			combined = tf.concat([frame0,frame1], 3)						#[N H W 6] 

		self.resnet = params.resnet
		
		self.flowScale = params.flowScale

		self.buildNetwork(combined, resnet = self.resnet)

	def buildNetwork(self, inputs, resnet):
		
		batchSize = inputs.get_shape()[0]

		conv = convLayerRelu(inputs, 7, 64, 2)								#[N H/2  W/2    64]

		if resnet:

			conv2   = resBlock(conv,    5,  128)							#[N H/4  W/4   128]
			conv3_1 = resBlock(conv2,   5,  256)							#[N H/8  W/8   256]
			conv4_1 = resBlock(conv3_1, 3,  512)							#[N H/16 W/16  512]
			conv5_1 = resBlock(conv4_1, 3,  512)							#[N H/32 W/32  512]
			conv6_1 = resBlock(conv5_1, 3, 1024)							#[N H/64 W/64 1024]					

		else:
						
			conv2   = convLayerRelu(conv,    5,  128, 2)					#[N H/4  W/4   128]

			conv3   = convLayerRelu(conv2,   5,  256, 2)					#[N H/8  W/8   256]
			conv3_1 = convLayerRelu(conv3,   3,  256, 1)					#[N H/8  W/8   256]

			conv4   = convLayerRelu(conv3_1, 3,  512, 2)					#[N H/16 W/16  512]
			conv4_1 = convLayerRelu(conv4,   3,  512, 1)					#[N H/16 W/16  512]

			conv5   = convLayerRelu(conv4_1, 3,  512, 2)					#[N H/32 W/32  512]
			conv5_1 = convLayerRelu(conv5,   3,  512, 1)					#[N H/32 W/32  512]

			conv6   = convLayerRelu(conv5_1, 3, 1024, 2)					#[N H/64 W/64 1024]
			conv6_1 = convLayerRelu(conv6,   3, 1024, 1)					#[N H/64 W/64 1024]

		fcInput = tf.reshape(conv6_1, (batchSize, -1))

		fc2 = fullyConnectedLayerRelu(fcInput, 1024)
		fc3 = fullyConnectedLayerRelu(fc2,     1024)
		fc4 = fullyConnectedLayerRelu(fc3,     2048)
		fc5 = fullyConnectedLayerRelu(fc4,     4096)

		fcOutput = tf.reshape(fc5, (batchSize, 4, 8, -1))

		with tf.variable_scope(None, default_name = "predict_flow6"):
			
			predict_flow6 = convLayer(fcOutput, 3, 2, 1)						#[N H/64 W/64    2]

		deconv5 = deconvLayerRelu(conv6_1, 4, 512, 2)						#[N H/32 W/32  512]

		concat5 = flowRefinementConcat(deconv5, conv5_1, predict_flow6)		#[N H/32 W/32 1026]
		
		with tf.variable_scope(None, default_name = "predict_flow5"):
			
			predict_flow5 = convLayer(concat5, 3, 2, 1)						#[N H/32 W/32    2]

		deconv4 = deconvLayerRelu(concat5, 4, 256, 2)						#[N H/16 W/16  256]
		
		concat4 = flowRefinementConcat(deconv4, conv4_1, predict_flow5)		#[N H/16 W/16  770]
		
		with tf.variable_scope(None,default_name="predict_flow4"):
		
			predict_flow4 = convLayer(concat4, 3, 2, 1)						#[N H/16 W/16    2]

		deconv3 = deconvLayerRelu(concat4, 4, 128, 2)						#[N H/8  W/8   128]
		
		concat3 = flowRefinementConcat(deconv3, conv3_1, predict_flow4)		#[N H/8  W/8   386]
		
		with tf.variable_scope(None,default_name="predict_flow3"):
		
			predict_flow3 = convLayer(concat3, 3, 2, 1)						#[N H/8  W/8     2]

		deconv2 = deconvLayerRelu(concat3, 4, 64, 2)						#[N H/4  W/4    64]
		
		concat2 = flowRefinementConcat(deconv2, conv2, predict_flow3)		#[N H/4  W/4   194]
		
		with tf.variable_scope(None,default_name="predict_flow2"):
		
			predict_flow2 = convLayer(concat2, 3, 2, 1)						#[N H/4  W/4     2]

		deconv1 = deconvLayerRelu(concat2, 4, 32, 2)						#[N H/2  W/2    32]
		
		concat1 = flowRefinementConcat(deconv1, conv, predict_flow2)		#[N H/2  H/2    98]
		
		with tf.variable_scope(None,default_name="predict_flow1"):
		
			predict_flow1 = convLayer(concat1, 3, 2, 1)						#[N H/2  H/2     2]

		deconv0 = deconvLayerRelu(concat1, 4, 16, 2)						#[N H    W      16]
		
		concat0 = flowRefinementConcat(deconv0, inputs, predict_flow1)		#[N H    W      24]
		
		with tf.variable_scope(None,default_name="predict_flow0"):
		
			predict_flow0 = convLayer(concat0, 3, 2, 1)						#[N H    W       2]

		predict_flow0 = predict_flow0 * self.flowScale						#[N H    W       2]

		self.flow0 = predict_flow0

		self.flows = [predict_flow0]

class TrainingLoss:

	def __init__(self, params, networkF, networkB, trainingData, recLossBWeight):

		with tf.variable_scope(None, default_name = "TrainingLoss"):
			
			weightDecay = params.weightDecay

			predFlowF = networkF.flows[0]
			predFlowB = networkB.flows[0]
			
			frame0 = trainingData.frame0
			frame1 = trainingData.frame1
			
			vpm = trainingData.validMask

			recLossF, endPointErrorF, ssimF, outImgF, epe2F, occInvalidMaskF = unsupFlowLoss(predFlowF, frame0, frame1, vpm, params)

			if params.lossComponentsBackward:
				
				recLossB, endPointErrorB, ssimB, outImgB, epe2B, occInvalidMaskB = unsupFlowLoss(predFlowB, frame1, frame0, vpm, params)

			with tf.variable_scope(None,default_name="weightDecay"):
				
				weightLoss = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection("weights")])) * weightDecay

			self.recLossBWeight = recLossBWeight

			self.flowForward = flowToRgb(predFlowF)

			self.flowBackward = flowToRgb(predFlowB)

			#consistencyLoss = consistency(predFlowF * occInvalidMaskF * occInvalidMaskB, predFlowB * occInvalidMaskF * occInvalidMaskB)

			if params.lossComponentsBackward:
				
				totalLoss = recLossF * (1.0 - self.recLossBWeight) + recLossB * self.recLossBWeight  + weightLoss 	

			else:

				totalLoss = recLossF + weightLoss 

			if params.lossComponentsBackward:

				endPointError = (endPointErrorB + endPointErrorF) / 2

				ssimError = (ssimF + ssimB) / 2

				epe2Error = (epe2F + epe2B) / 2

			else:

				endPointError = endPointErrorF

				ssimError = ssimF

				epe2Error = (epe2F + epe2B) / 2

			self.loss = totalLoss

			self.epe = endPointError

			self.ssimloss = ssimError

			self.img0 = outImgF

			self.img1 = outImgB

			self.epe2Cost = epe2Error