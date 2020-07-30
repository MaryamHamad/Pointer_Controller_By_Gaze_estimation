'''
Load, pre-process input and output for Face detection model
'''

import os
import cv2
import logging as log
from openvino.inference_engine import IECore

# Identify default confidence threshold
CONFIDANCE_THRESHOLD = 0.6

class ModelFaceDetection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self):
        '''
        Set instance variables.
        '''
        self.core = None
        self.exec_network = None
        self.device = None
        self.model = None
        self.model_structure = None
        self.model_weights =None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None

    def load_model(self, model_name, device='CPU'):
        '''
        Load the model to the specified device.
        Return excutable network
        '''
        try:
            self.device = device
            self.model = model_name
            self.model_structure = model_name
            self.model_weights = os.path.splitext(self.model_structure)[0] + ".bin"

            #Plugin initialization
            self.core = IECore()
            try:
                self.model = IECore().read_network(self.model_structure, self.model_weights)
            except Exception as e:
                raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
            self.input_blob = next(iter(self.model.inputs))
            self.input_shape = self.model.inputs[self.input_blob].shape
            self.output_blob = next(iter(self.model.outputs))
            self.output_shape = self.model.outputs[self.output_blob].shape

            # Network loding into core plugin
            self.exec_network = self.core.load_network(network = self.model, device_name = self.device)
        except Exception as e:
            log.error("Error in loading face detection model:" + str(e), exc_info=True)
        return self.exec_network

    def predict(self, frame):
        '''
        Run predictions on the input image.
        '''
        # Image preprocessing
        p_frame = self.preprocess_input(frame, self.input_shape)

        try:
            # Asynchronous inferece starting
            self.exec_network.start_async(request_id=0, inputs={self.input_blob: p_frame})

            # Waiting and getting the inference results
            if self.exec_network.requests[0].wait(-1) == 0:
                outputs = self.exec_network.requests[0].outputs[self.output_blob]

                # Output preprocessing and returen face coordinates
                coordinates = self.preprocess_output(frame, outputs)
        except Exception as e:
            log.error("Couldn't run inference requist on face detection model due to these errors: " + str(e), exc_info=True)
        return coordinates

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, frame, input_shape):
        '''
        Preprocess the input image
        '''
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, frame, outputs):
        '''
        Preprocess the output before feeding it to the next model and return
        the output cropped face and face coordenates
        '''
        w, h = frame.shape[1], frame.shape[0]
        coords = []
        for box in outputs[0][0]:
            c = box[2]
            if c >= CONFIDANCE_THRESHOLD:
                x1 = int(box[3] * w)
                y1 = int(box[4] * h)
                x2 = int(box[5] * w)
                y2 = int(box[6] * h)
                coords.extend((x1, y1, x2,y2))
        return coords