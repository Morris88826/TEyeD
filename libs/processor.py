from collections import defaultdict
import glob
import os
from re import L
import cv2
import time
import json
class Processor():
    def __init__(self, video_path, data_dir="./sample_data", process_video=True) -> None:
        self.video_path = video_path
        self.video_name = video_path.split('/')[-1]
        self.annotations_path = sorted(glob.glob(data_dir+'/annotations/{}*.txt'.format(self.video_name)))
        self.data_dir = data_dir
        self.annotations_dir = self.data_dir + '/annotations'

        with open('{}/{}eye_ball.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()
        self.num_frames = int(lines[-1].split(';')[0])

        self.out_root = './processed/{}'.format(self.video_name.replace('.mp4', ''))
        if not os.path.exists('./processed'):
            os.mkdir('./processed')
        if not os.path.exists(self.out_root):
            os.mkdir(self.out_root)
        self.info = defaultdict(dict)


        if not os.path.exists(self.out_root+'/info.json'):
            self.generate_info(process_video=process_video)
        
        # self.images_path = sorted(glob.glob(self.out_root+'/image/*'), key=lambda s: int(s.split('/')[-1].split('.')[0]))
        self.images_path = sorted(glob.glob(self.out_root+'/image/*'))
        with open(self.out_root+'/info.json', 'r') as jsonfile:
            self.info = json.load(jsonfile)

    def get_data(self):
        return self.info, self.images_path


    def generate_info(self, process_video=True):
        if process_video:
            self.process_video()
        for annotation_path in self.annotations_path:
            self.annotation_type = annotation_path.split('/')[-1].replace(self.video_name, '').replace('.txt', '')
            if self.annotation_type == "eye_ball":
                self.process_eye_ball()
            elif self.annotation_type == "eye_movements":
                # self.process_eye_movements()
                pass
            elif self.annotation_type == "gaze_vec":
                self.process_gaze_vec()
            elif self.annotation_type == "iris_eli":
                self.process_iris_eli()
            elif self.annotation_type == "iris_lm_2D":
                self.process_iris_lm_2D()
            elif self.annotation_type == "iris_lm_3D":
                # self.process_iris_lm_3D()
                pass
            elif self.annotation_type == "lid_lm_2D":
                self.process_lid_lm_2D()
            elif self.annotation_type == "lid_lm_3D":
                # self.process_lid_lm_3D()
                pass
            elif self.annotation_type == "pupil_eli":
                self.process_pupil_eli()
            elif self.annotation_type == "pupil_in_iris_eli":
                self.process_pupil_in_iris_eli()
            elif self.annotation_type == "pupil_lm_2D":
                self.process_pupil_lm_2D()
            elif self.annotation_type == "pupil_lm_3D":
                # self.process_pupil_lm_3D()
                pass
            elif self.annotation_type == "validity_iris":
                self.process_validity_iris()
            elif self.annotation_type == "validity_lid":
                self.process_validity_lid()
            elif self.annotation_type == "validity_pupil":
                self.process_validity_pupil()
            else:
                raise NotImplementedError
        
        with open(self.out_root+'/info.json', 'w') as jsonfile:
            json.dump(self.info, jsonfile, indent=4)
    
    def process_video(self):
        #print("---------------------")
        #print("Processing Video ~")
        images_out_root = self.out_root+'/image'
        if not os.path.exists(images_out_root):
            os.mkdir(images_out_root)

        vidcap = cv2.VideoCapture(self.video_path)
        self.fps = vidcap.get(cv2.CAP_PROP_FPS)
        
        success,image = vidcap.read()
        count = 1
        while success:
            cv2.imwrite("{}/{:07d}.png".format(images_out_root, count), image)     # save frame as JPEG file      
            try:
                success, image = vidcap.read()
                count += 1
            except:
                success = False
        #print("Done.")

    def process_eye_ball(self):
        #print("---------------------")
        #print("Processing Eye Ball ~")
        with open('{}/{}eye_ball.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()
        
        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            radius = float(_info[1])
            x = float(_info[2])
            y = float(_info[3])
            z = float(_info[4])


            self.info[frame_id]["eye_ball"] = {
                "radius": radius,
                "center": [x,y,z]
            }
        
        #print("Done.")


    def process_eye_movements(self):
        #print("---------------------")
        #print("Processing Eye Movements ~")
        with open('{}/{}eye_movements.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()


    def process_gaze_vec(self):
        #print("---------------------")
        #print("Processing Gaze Vec ~")
        with open('{}/{}gaze_vec.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()
        
        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            x = float(_info[1])
            y = float(_info[2])
            z = float(_info[3])

            self.info[frame_id]["gaze_vec"] = {
                "vector": [x,y,z]
            }

        #print("Done.")

    def process_iris_eli(self): #elipse
        #print("---------------------")
        #print("Processing Iris Eli ~")

        with open('{}/{}iris_eli.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
                    lines = f.readlines()
        

        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            angle = float(_info[1])
            center_x = float(_info[2])
            center_y = float(_info[3])
            width = float(_info[4])
            height = float(_info[5])


            self.info[frame_id]["iris_eli"] = {
                "angle": angle,
                "center": [center_x, center_y],
                "width": width,
                "height": height
            }
        #print("Done.")

    def process_iris_lm_2D(self):
        #print("---------------------")
        #print("Processing Iris Lm 2D ~")

        with open('{}/{}iris_lm_2D.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
                    lines = f.readlines()

        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            _landmarks = _info[2:-1]

            landmarks = []
            for i in range(0, len(_landmarks), 2):
                x = float(_landmarks[i])
                y = float(_landmarks[i+1])
                landmarks.append([x,y])

            self.info[frame_id]['iris_lm_2D'] = {
                "landmarks": landmarks
            }
        #print("Done.")

    def process_iris_lm_3D(self):
        #print("---------------------")
        #print("Processing Iris Lm 3D ~")
        pass

    def process_lid_lm_2D(self):
        #print("---------------------")
        #print("Processing Lid Lm 2D ~")

        with open('{}/{}lid_lm_2D.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()
        
        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            _landmarks = _info[2:-1]

            landmarks = []
            for i in range(0, len(_landmarks), 2):
                x = float(_landmarks[i])
                y = float(_landmarks[i+1])
                landmarks.append([x,y])

            self.info[frame_id]['lid_lm_2D'] = {
                "landmarks": landmarks
            }
        #print("Done.")

    def process_lid_lm_3D(self):
        #print("---------------------")
        #print("Processing Lid Lm 3D ~")
        pass

    def process_pupil_eli(self):
        #print("---------------------")
        #print("Processing Pupil Eli ~")
        with open('{}/{}pupil_eli.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()
        

        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            angle = float(_info[1])
            center_x = float(_info[2])
            center_y = float(_info[3])
            width = float(_info[4])
            height = float(_info[5])


            self.info[frame_id]["pupil_eli"] = {
                "angle": angle,
                "center": [center_x, center_y],
                "width": width,
                "height": height
            }
        #print("Done.")
    def process_pupil_in_iris_eli(self):
        #print("---------------------")
        #print("Processing Pupil In Iris Eli")
        with open('{}/{}pupil_in_iris_eli.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()
        

        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            angle = float(_info[1])
            center_x = float(_info[2])
            center_y = float(_info[3])
            width = float(_info[4])
            height = float(_info[5])


            self.info[frame_id]["pupil_in_iris_eli"] = {
                "angle": angle,
                "center": [center_x, center_y],
                "width": width,
                "height": height
            }
        #print("Done.")
    def process_pupil_lm_2D(self):
        #print("---------------------")
        #print("Processing Pupil Lm 2D")

        with open('{}/{}pupil_lm_2D.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()
        
        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            _landmarks = _info[2:-1]

            landmarks = []
            for i in range(0, len(_landmarks), 2):
                x = float(_landmarks[i])
                y = float(_landmarks[i+1])
                landmarks.append([x,y])

            self.info[frame_id]['pupil_lm_2D'] = {
                "landmarks": landmarks
            }
        #print("Done.")

    def process_pupil_lm_3D(self):
        #print("---------------------")
        #print("Processing Pupil Lm 3D")
        pass

    def process_validity_iris(self):
        #print("---------------------")
        #print("Processing Validity Iris")

        with open('{}/{}validity_iris.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()
        
        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            valid = int(_info[1])

            self.info[frame_id]['validity_iris'] = True if valid>0 else False
        #print("Done.")

    def process_validity_lid(self):
        #print("---------------------")
        #print("Processing Validity Lid")
        with open('{}/{}validity_lid.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()
        
        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            valid = int(_info[1])

            self.info[frame_id]['validity_lid'] = True if valid>0 else False
        #print("Done.")

    def process_validity_pupil(self):
        #print("---------------------")
        #print("Processing Validity Pupil")
        with open('{}/{}validity_pupil.txt'.format(self.annotations_dir, self.video_name), 'r') as f:
            lines = f.readlines()
        
        for l in lines[1:]:
            _info = l.split(';')
            frame_id = int(_info[0])
            valid = int(_info[1])

            self.info[frame_id]['validity_pupil'] = True if valid>0 else False
        #print("Done.")