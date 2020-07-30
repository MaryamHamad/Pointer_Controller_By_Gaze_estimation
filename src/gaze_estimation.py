import os
import cv2
import logging as log
from openvino.inference_engine import IECore

class ModelGazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''

    def __init__(self):
        self.core = None
        self.exec_network = None
        self.device = None
        self.model = None
        self.model_structure = None
        self.model_weights = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None

    def load_model(self, model_name, device='CPU'):
        '''
        Load the model to the specified device.
        Return executable nework
        '''
        try:
            self.device = device
            self.model = model_name
            self.model_structure = model_name
            self.model_weights = os.path.splitext(self.model_structure)[0] + ".bin"

            #Plugin initialization
            self.core = IECore()
            try:
                #read the model
                self.model = self.core.read_network(self.model_structure, self.model_weights)
            except Exception as e:
                raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

            # get eye shape which is same to both eyes
            self.input_shape = self.model.inputs['left_eye_image'].shape
            self.input_blob = [n for n in self.model.outputs.keys()]

            # Network loding into core
            self.exec_network = self.core.load_network(network = self.model, device_name = self.device)
        except Exception as e:
            log.error("Error in loading gaze estimation model:" + str(e), exc_info=True)
        return self.exec_network

    def predict(self,l_eye, r_eye, pose_vector):
        '''
        Run predictions on the input image and return the gaze vector.
        '''
        try:
            # Asynchronous inferece starting
            self.exec_network.start_async(request_id=0, inputs={'left_eye_image': self.preprocess_input(l_eye, self.input_shape),
                                                                'right_eye_image': self.preprocess_input(r_eye, self.input_shape),
                                                                'head_pose_angles': pose_vector})
            # Waiting and getting the inference results
            if self.exec_network.requests[0].wait(-1) == 0:
                results = self.exec_network.requests[0].outputs[self.input_blob[0]]
                gaze_output = self.preprocess_output(results)
        except Exception as e:
            log.error("Couldn't run inference requist on gaze estimation model due to these errors: " + str(e), exc_info=True)
        return gaze_output

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image,input_shape):
        '''
        Preprocess the input image
        '''
        p_frame = cv2.resize(image, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame

    def preprocess_output(self, outputs):
        #return the gaze array
        return  outputs[0]
