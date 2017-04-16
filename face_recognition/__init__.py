from face import FaceRec

def main():
    f = FaceRec("/training_set","/test_set")
    f.run_training_set()
    f.run_testing_set()

if __name__ == "__main__":
    main()

