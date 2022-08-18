import time
import cv2
import sys
import dlib as dl
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ensemble_model import Net_Model

# Function to obtain the face boundaries

def face_coord(image):

    large_image = [0.2, 0.5]
    limit = (500 * 500)
    small_image = [0.5, 1.0]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    gray_size = gray.size

    if gray_size > limit:
        scale = large_image
    
    else:
        scale = small_image
    
    for each in scale:

        output = []

        transform = cv2.resize(image, None, fx=each, fy=each)

        # The dlib CNN face detection model is loaded

        cascade_method = dl.cnn_face_detection_model_v1('./model/dlib/cnn_face_detector.dat')

        final_face_data = cascade_method(transform)

        if final_face_data != None:

            for each_1 in final_face_data:

                left_data = each_1.rect.left()

                right_data = each_1.rect.right()

                top_data = each_1.rect.top()

                bottom_data = each_1.rect.bottom()

                output.append([[left_data, top_data], [right_data, bottom_data]])

        
            final_face_output = np.array(output)

            coord_len = len(final_face_output)

            scale_rat = (1 / each)

            # The threshold for the face region is intialized

            thresh = final_face_output * scale_rat

         
            if coord_len > 0:

                final_face_output = thresh.astype(int)
                
                break

    face_sum = np.sum(final_face_output[0])

    if coord_len > 0 and face_sum > 0:

        detect_output = final_face_output[0]
    
    else:

        detect_output = None

    return detect_output

# Function for the emotion categories

def category(class_val):

    # The discrete emotion categories are defined

    dict = {}

    dict[0] = 'Neutral'
    dict[1] = 'Happy'
    dict[2] = 'Sad'
    dict[3] = 'Surprise'
    dict[4] = 'Fear'
    dict[5] = 'Disgust'
    dict[6] = 'Anger'
    dict[7] = 'Contempt'

    val = dict[class_val]

    return val


# Function for emotion detection

def emotion_detect(detect_face, device):

    splice_1 = 0
    splice_2 = 1

    # The base net model is intitalized
    
    model = Net_Model(device)

    array_affec_1 = []
    array_affec_2 = []

    # The emotion data and the valence,arousal data are returned

    emotion_data, val_aro_data = model(detect_face)

    for each in val_aro_data:

        array_affec_1.append(each[0].cpu().detach().numpy())

    affect_data = np.array(array_affec_1)

    affect_num = (affect_data[:, splice_2] + splice_2) / 2.0


    affect_data[:, 1] = np.clip(affect_num, splice_1, splice_2)

    data_mean = np.mean(affect_data, splice_1)

    aff_data = np.expand_dims(data_mean, axis = splice_1)

    result_affect = np.concatenate((affect_data, aff_data), axis = splice_1)

    for each in emotion_data:

        array_affec_2.append(each[0].cpu().detach().numpy())
   
    res_data = np.array(array_affec_2)


    fin_res = np.zeros(res_data.shape[1])

    result_emotion = []

    for each in res_data:
        
        result_emotion.append(category(np.argmax(each)))

        fin_res[np.argmax(each)] += 1

    result_emotion.append(category(np.argmax(fin_res)))

    return result_emotion, result_affect


def emotion(image):

    if sys.argv[2]:

        device = torch.device("cpu")
    
    else:

        device = torch.device("cuda")

    loaded_image = cv2.imread(image, cv2.IMREAD_COLOR)


    output_face = face_coord(loaded_image)

    splice_1 = output_face[0][1]
    splice_2 = output_face[1][1]
    splice_3 = output_face[0][0]
    splice_4 = output_face[1][0]

    final_face = loaded_image[splice_1:splice_2, splice_3:splice_4, :]

    tranform = cv2.resize(final_face, (96, 96))

    tranform = Image.fromarray(tranform)

    # The input image is being transformed

    tranform = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                 std=[1.0, 1.0, 1.0])(transforms.ToTensor()(tranform)).unsqueeze(0)
    
    transform_face = tranform.to(device)

    emotion, affect = emotion_detect(transform_face, device)

    print("Emotion :", max(set(emotion), key = emotion.count))
    print("Valence :", affect[-1][0])
    print("Arousal :", affect[-1][1])


if __name__ == "__main__":

    # The image_path is the path to the image file

    image_path = sys.argv[1]
    emotion(image_path)