import face_recognition as fr
import cv2
import numpy as np
import os

def face_verification_from_image(img, dataset_dir="face_dataset"):
    known_face_encodings = []
    known_face_names = []
    face_locations = []
    face_encodings = []
    face_names = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".jpg"):
            try:
                file_dir = os.path.join(dataset_dir, file)
                name = file.split('.')[0]
                face_encoding = fr.face_encodings(fr.load_image_file(file_dir))[0]
                known_face_names.append(name)
                known_face_encodings.append(face_encoding)            
            except:
                pass
    image = cv2.imread(img)
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = fr.face_locations(rgb_small_frame)
    face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    return face_names

def get_frame(img):
    image = cv2.imread(img)
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    return rgb_small_frame

def verify_two_image(img1, img2):
    # params: 2 images
    # return: True if 2 images match
    try:
        frame1 = get_frame(img1)
        frame2 = get_frame(img2)
        face1_encoding = fr.face_encodings(frame1)[0]
        face2_encoding = fr.face_encodings(frame2)[0]
        results = fr.compare_faces([face1_encoding], face2_encoding)

        return results[0]
    except:
        return False
