import time
import sys
import logging as log
import cv2
from math import cos, sin, pi
import os
from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import ModelFaceDetection
from head_pose_estimation import ModelHeadPoseEstimation
from facial_landmarks_detection import ModelFacialLandmarksDetection
from gaze_estimation import ModelGazeEstimation

class PointerControllerApp:
    def get_arguments(self):
        '''
        Gets the input arguments
        '''
        parser = ArgumentParser("Control the mouse pointer from a stream by estimating eyes dirction")

        # Identify helper to each command
        i_helper = "Insert input path, it can be image, video and webcam. In case if webcam please insert 'CAM'."
        fm_helper = "Insert thhe location of the face detection model .xml file."
        hm_helper = "Insert thhe location of the head pose estimation model .xml file."
        flm_helper = "Insert the location of the facial landmarks detection model .xml file."
        gm_helper = "Insert the location of the gaze estimation model .xml file."
        d_helper = "Insert device name, it can be CPU, GPU, FPGA, VPU."
        df_helper = "Flag to enable inference outputs displaying"

        # input stream
        parser.add_argument("-i", "--input", required=True, type=str,help = i_helper)

        # models
        parser.add_argument("-fm", "--face_detection_model", required=True, help = fm_helper)
        parser.add_argument("-hm", "--head_pose_estimation_model", required=True, help = hm_helper)
        parser.add_argument("-flm", "--facial_landmarks_detection_model", required=True, help = flm_helper)
        parser.add_argument("-gm", "--gaze_estimation_model", required=True, help = gm_helper)

        # device and model output display settings
        parser.add_argument("-d", "--device", required=False, default="CPU", help = d_helper)
        parser.add_argument("-df", "--display_flag", required=False, default=True,help= df_helper)
        args = parser.parse_args()

        return args


    def infer(self, args):
        # Create instances from the models' classes
        FDM_net = ModelFaceDetection()
        HPE_net = ModelHeadPoseEstimation()
        FLD_net = ModelFacialLandmarksDetection()
        GEM_net = ModelGazeEstimation()
        mouse_controller = MouseController('high', 'fast')

        # Load the models
        start1 = time.time()
        FDM_net.load_model(args.face_detection_model, args.device)
        FDM_load_t = time.time() - start1

        start2 = time.time()
        HPE_net.load_model(args.head_pose_estimation_model, args.device)
        HPE_load_t = time.time() - start2

        start3 = time.time()
        FLD_net.load_model(args.facial_landmarks_detection_model, args.device)
        FLD_load_t = time.time() - start3

        start4 = time.time()
        GEM_net.load_model(args.gaze_estimation_model, args.device)
        GEM_load_t = time.time() - start4

        print('All models are loaded!')

        #Check the inputs
        # To make the mouse moving we need video stream either from camera or video path
        if args.input.lower() == 'cam':
            # Initialise the InputFeeder class
            input_feeder = InputFeeder(input_type= 'cam', input_file=args.input)
        else:
            if not os.path.isfile(args.input):
                log.error("Please insert valid video path to run the app.")
                exit()
            # Initialise the InputFeeder class
            input_feeder = InputFeeder(input_type='video', input_file=args.input)

        # Load the video capture
        input_feeder.load_data()

        # Inference time
        inference = time.time()

        # Read from the video capture
        for flag, frame in input_feeder.next_batch():
            if not flag:
                break
            key_pressed = cv2.waitKey(60)

            # Run inference on the models
            start5 = time.time()
            face_coords = FDM_net.predict(frame)
            FDM_infer_t = time.time()-start5

            # crop the face from the frame
            cropped_face = frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]

            #Everything depends on the face detection output, if no face detected then repeat
            if len(face_coords) == 0:
                log.error("There is no faces detected.")
                continue
            start6 = time.time()
            HP_angles = HPE_net.predict(cropped_face, face_coords)
            HPE_infer_t = time.time() - start6

            if args.display_flag:
                #### display the face
                O_frame = cv2.rectangle(frame.copy(), (face_coords[0], face_coords[1]),
                                          (face_coords[2], face_coords[3]), (255, 255, 0), 2)

                #### display the pose angles
                # Link for pose estimation output code resource: https://sudonull.com/post/6484-Intel-OpenVINO-on-Raspberry-Pi-2018-harvest
                cos_r = cos(HP_angles[2] * pi / 180)
                sin_r = sin(HP_angles[2] * pi / 180)
                cos_y = cos(HP_angles[0] * pi / 180)
                sin_y = sin(HP_angles[0] * pi / 180)
                cos_p = cos(HP_angles[1] * pi / 180)
                sin_p = sin(HP_angles[1] * pi / 180)

                x = int((face_coords[0] + face_coords[2]) / 2)
                y = int((face_coords[1] + face_coords[3]) / 2)
                cv2.line(O_frame, (x, y), (x + int(65 * (cos_r * cos_y + sin_y * sin_p * sin_r)), y + int(65 * cos_p * sin_r)),(255, 0, 0), thickness=2)
                cv2.line(O_frame, (x, y), (x + int(65 * (cos_r * sin_y * sin_p + cos_y * sin_r)), y - int(65 * cos_p * cos_r)), (0, 255, 0), thickness=2)
                cv2.line(O_frame, (x, y), (x + int(65 * sin_y * cos_p), y + int(65 * sin_p)), (0, 0, 255), thickness=2)

            start7 = time.time()
            l_e, r_e, l_e_image, r_e_image, e_center = FLD_net.predict(O_frame, cropped_face, face_coords)
            FLD_infer_t = time.time() - start7

            ###display landmarks for both eyes
            if args.display_flag:
                cv2.circle(O_frame, (face_coords[0] + l_e[0], face_coords[1] + l_e[1]), 29, (0, 255, 255), 2)
                cv2.circle(O_frame, (face_coords[0] + r_e[0], face_coords[1] + r_e[1]), 29, (0, 255, 255), 2)


            start8 = time.time()
            g_vec = GEM_net.predict(l_e_image, r_e_image, HP_angles)
            GEM_infer_t = time.time() - start8

            ###display gaze model output
            if args.display_flag:
                cv2.arrowedLine(O_frame, (int(e_center[0][0]), int(e_center[0][1])), (int(e_center[0][0]) + int(g_vec[0] * 90), int(e_center[0][1]) + int(-g_vec[1] * 90)),  (203,192,255), 2)
                cv2.arrowedLine(O_frame, (int(e_center[1][0]), int(e_center[1][1])), (int(e_center[1][0]) + int(g_vec[0] * 90), int(e_center[1][1]) + int(-g_vec[1] * 90)), (203, 192, 255), 2)

            # change the pointer position according to the estimated gaze direction
            mouse_controller.move(g_vec[0], g_vec[1])

            if key_pressed == 27:
                break

            # Display the resulting frame
            cv2.imshow('Mouse Controller App Results', cv2.resize(O_frame, (750, 550)))

        inference_time = time.time() - inference

        print("Loading time: \n1-Face detection: " + str(FDM_load_t)+ "\n2- Head pose estimation: " + str(HPE_load_t)+ "\n3-Facial landmarks model: " + str(FLD_load_t)+ "\n4-Gaze estimation model:  " + str(GEM_load_t))
        print("Output inference time: \n1-Face detection: " + str(FDM_infer_t)+ "\n2- Head pose estimation: " + str(HPE_infer_t)+ "\n3-Facial landmarks model: " + str(FLD_infer_t)+ "\n4-Gaze estimation model:  " + str(GEM_infer_t))

        # close the input feeder and destroy all opened windows
        input_feeder.close()
        cv2.destroyAllWindows


    def main(self, args):
        # Run inference on the stream
        self.infer(args)

if __name__ == '__main__':
    app = PointerControllerApp()
    args = app.get_arguments()
    app.main(args)
