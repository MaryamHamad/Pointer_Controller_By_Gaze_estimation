# Pointer_Controller_By_Gaze_estimation
Change computer mouse position based on gaze estimation 

About the project:
In this project, a mouse pointer is controlled by the gaze estimations. Basically, 4 openVino models are used in this project to achieve the objective and they are listed as following: 
* face detection model (https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* facial landmark model (https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* gaze estimation model (https://docs.openvinotoolkit.org/2019_R1/_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)
* head pose estimation model(https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
The main objective is to use locally different models to change the mouse position. 

## Project Set-up and Installation
project directory structure: image can be found in project images project_directory_structure.png
Project inference pipeline: image can be found in project images inference_pipeline.png

Install the dependencies that the project requires:
1- install OpenVINOtoolkit from (https://docs.openvinotoolkit.org/latest/index.html) and set the environment variables by running: source /opt/intel/openvino/bin/setupvars.sh

2- use a virtual environment to isolate the project dependencies (more information can be found here https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
you can use conda or virtualenv for example:  pip install virtualenv

3- you need to creat virtual enviroment via virtualenv and activate it by running:  
	virtualenv pointer-controller  
	source computer-pointer-controller/bin/activate 

4- you need to install project dependencies by moving to the pointer-controller directory and then running: pip install -r requirements.txt 

5- create a models directory, cd into it and download the four models such as: 
	sudo /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001

6- run the code by using the following codes in the Demo section.


## Demo
To run a basic demo:

Python3 src/main.py -fm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -hm models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -flm models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -gm models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i inputs/demo.mp4

## Documentation
usage: project arguments:
-fm: FACE_DETECTION_MODEL: Path of the face-detection model (required).
-flm: FACIAL_LANDMARKS_DETECTION_MODEL: Path of the landmark-regression model (required).
-hm: HEAD_POSE_ESTIMATION_MODEL: Path of the head-pose-estimation model (required).
-gm: GAZE_ESTIMATION_MODEL: The location of the gaze-estimation mode (required).
-i: INPUT: Input type can be 'CAM' or give image or video.
-d: DEVICE: The device name, if not 'CPU', can be GPU, FPGA or MYRIAD.
-df :DISPLAY_FLAG: To display the model's output by drawing boxes or outputs on the frame 

## Benchmarks
using CPU, and openVINOtoolkit on macOS, I got these results:
--------



#Loading time:

FP32:
	1-Face detection: 0.405
	2- Head pose estimation: 0.287
	3-Facial landmarks model: 0.149
	4-Gaze estimation model:  0.163
--------
FP16:
	1-Face detection: 0.405
	2- Head pose estimation: 0.274
	3-Facial landmarks model: 0.132
	4-Gaze estimation model:  0.155
--------
INT8: 
	1-Face detection: 0.446
	2- Head pose estimation: 0.424
	3-Facial landmarks model: 0.182
	4-Gaze estimation model:  0.310
--------


#Inference time:

FP32: 
	1-Face detection: 0.0155
	2- Head pose estimation: 0.0014
	3-Facial landmarks model: 0.00123
	4-Gaze estimation model:  0.00163
----------
FP16: 
	1-Face detection: 0.0153
	2- Head pose estimation: 0.0015
	3-Facial landmarks model: 0.0013
	4-Gaze estimation model:  0.0018
----------
INT8:
	1-Face detection: 0.0156
	2- Head pose estimation: 0.00128
	3-Facial landmarks model: 0.00126
	4-Gaze estimation model:  0.00132
## Results
I got slight differences for different precision, higher precision models give better accuracy and need more inference time in total, however, to achieve a faster inferencing lower precision models can be used. From the results it seems the face detection is the most time consuming among the 4 models and all the models relies on its outputs.
