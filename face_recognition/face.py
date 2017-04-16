#imports
import sys,os,dlib,glob,numpy
from skimage import io

class FaceRec:
    def __init__(self,faces_folder_path,img_path):
        self.BASE_DIR = os.path.join(os.path.dirname(__file__))
        self.faces_folder_path   = self.BASE_DIR + faces_folder_path
        self.img_path            = self.BASE_DIR + img_path
        self.detector            = dlib.get_frontal_face_detector()
        self.sp                  = dlib.shape_predictor(self.BASE_DIR+"/training_model/shape_predictor_68_face_landmarks.dat")
        self.face_rec            = dlib.face_recognition_model_v1(self.BASE_DIR + "/training_model/dlib_face_recognition_resnet_model_v1.dat")

        # training
        self.descriptors = []

    def run_training_set(self):
        # train each of the training set
        for f in glob.glob(os.path.join(self.faces_folder_path, "*.jpg")):
            print("Processing file: {}".format(f))
            img = io.imread(f)
            #check the face
            dets = self.detector(img, 1)
            print("Number of faces detected: {}".format(len(dets)))
            for k, d in enumerate(dets):
                #checking each of the face point
                shape = self.sp(img, d)
                #get the face descriptor 1208D
                face_descriptor = self.face_rec.compute_face_descriptor(img, shape)
                v = numpy.array(face_descriptor)
                self.descriptors.append(v)


    def run_testing_set(self):
        for f in glob.glob(os.path.join(self.img_path, "*.jpg")):
            print("Processing file: {}".format(f))
            img = io.imread(f)
            # check the face
            dets = self.detector(img, 1)
            dist = []
            for k, d in enumerate(dets):
                # checking each of the face point
                shape = self.sp(img, d)
                # get the face descriptor 1208D
                face_descriptor = self.face_rec.compute_face_descriptor(img, shape)

                d_test = numpy.array(face_descriptor)

                #Euclidean distance
                for i in self.descriptors:
                    dist_ = numpy.linalg.norm(i - d_test)
                    dist.append(dist_)

            # the candidate name
            candidate = ['Shishi', 'Shishi', 'Bingbing']
            # make a dict of candidate and the distance
            c_d = dict(zip(candidate, dist))
            # sort the distance
            cd_sorted = sorted(c_d.iteritems(), key=lambda d: d[1])
            print "\nThe person is: ", cd_sorted[0][0]

            dlib.hit_enter_to_continue()

def main():
    program = FaceRec("/training_set","/test_set")
    program.run_training_set()
    program.run_testing_set()


if __name__ == "__main__":
    main()




