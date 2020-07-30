import os
import cv2
import logging as log
from openvino.inference_engine import IECore


class ModelFacialLandmarksDetection:
    '''
    Class for the Facial Landmark Detection Model.
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
            self.output_blob = next(iter(self.model.outputs))

            # Network loding into core plugin
            self.exec_network = self.core.load_network(network = self.model, device_name = self.device)
        except Exception as e:
            log.error("Error in loading facial-landmarks model:" + str(e), exc_info=True)
        return self.exec_network

    def predict(self, image, cropped_face, f_coords):
        '''
        Run predictions on the input image and return the left and right eyes and their centers.
        '''

        # Image preprocessing
        p_frame = self.preprocess_input(cropped_face, self.input_shape)
        try:
            # Asynchronous inferece starting
            self.exec_network.start_async(request_id=0, inputs={self.input_blob: p_frame})

            # Waiting and getting the inference results
            if self.exec_network.requests[0].wait(-1) == 0:
                results = self.exec_network.requests[0].outputs[self.output_blob]
                l_e, r_e, l_e_image, r_e_image, e_centers = self.preprocess_output(image, results, f_coords)
        except Exception as e:
            log.error("Couldn't run inference requist on faical landmarks model due to these errors: " + str(e), exc_info=True)
        return l_e, r_e, l_e_image, r_e_image, e_centers

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image, input_shape):
        '''
        Preprocess the input image
        '''
        p_frame = cv2.resize(image, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame

    def preprocess_output(self, image, outputs, f_coordinates):
        '''
        Note that: the net output is a blob with the shape: [1, 10], containing a row-vector of 10 floating point values
        for five Facial Landmarks (fl) coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
        '''
        fl = outputs.reshape(1, 10)[0]
        h = f_coordinates[3] - f_coordinates[1]  # face height
        w = f_coordinates[2] - f_coordinates[0] #face width
        l_e_coords = [int(fl[0] * w), int(fl[1] * h)]
        r_e_coords = [int(fl[2] * w), int(fl[3] * h)]

        #Compute eyes centers to use in in gaze estimation model
        centers = [[f_coordinates[0] + l_e_coords[0], f_coordinates[1] + l_e_coords[1]], [f_coordinates[0] + r_e_coords[0], f_coordinates[1] + r_e_coords[1]]]

        #Crop left eye from the face
        l_e_image = image[l_e_coords[1]+f_coordinates[1]  - 25: l_e_coords[1]+f_coordinates[1] + 25 , l_e_coords[0]+f_coordinates[0]  - 25: l_e_coords[0]+f_coordinates[0] + 25]
        # # Crop right eye from the face
        r_e_image = image[r_e_coords[1]+f_coordinates[1] - 25:f_coordinates[1] + r_e_coords[1] + 25 , f_coordinates[0] + r_e_coords[0] - 25:f_coordinates[0] + r_e_coords[0] + 25]
        return l_e_coords, r_e_coords, l_e_image, r_e_image, centers

