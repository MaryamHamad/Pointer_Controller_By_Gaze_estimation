import os
import cv2
import logging as log
from openvino.inference_engine import IECore

class ModelHeadPoseEstimation:
    '''
    Class for the Head Pose Estiamtion Model.
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
        self.model_weights = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None

    def load_model(self, model_name, device):
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
                self.model = self.core.read_network(self.model_structure, self.model_weights)
            except Exception as e:
                raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

            self.input_blob = next(iter(self.model.inputs))
            self.input_shape = self.model.inputs[self.input_blob].shape

            # Network loding into core plugin
            self.exec_network = self.core.load_network(network = self.model, device_name = self.device)

        except Exception as e:
            log.error("Error in loading head pose detection model:" + str(e), exc_info=True)

        return self.exec_network

    def predict(self, cropped_face, f_coords):
        '''
        Run predictions on the input image and return the head pose angles.
        '''
        # Image preprocessing
        p_frame = self.preprocess_input(cropped_face, self.input_shape)
        try:
            # Asynchronous inferece starting
            self.exec_network.start_async(request_id=0, inputs={self.input_blob: p_frame})

            # Waiting and getting the inference results
            if self.exec_network.requests[0].wait(-1) == 0:
                outputs = self.exec_network.requests[0].outputs
                hp_angles = self.preprocess_output(outputs, f_coords)
        except Exception as e:
            log.error("Couldn't run inference requist on head pose estimation model due to these errors: " + str(e), exc_info=True)
        return  hp_angles

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

    def preprocess_output(self, outputs, f_coords):
        '''
        Preprocess the output before feeding it to the next model.
        Return the outputs in head pose angels : [yaw, pitch, roll]
        '''
        return [outputs['angle_y_fc'][0][0], outputs['angle_p_fc'][0][0], outputs['angle_r_fc'][0][0]]