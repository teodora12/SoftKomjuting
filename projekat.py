import numpy as np
import os
from keras.models import model_from_json
import cv2
import matplotlib
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
from centroidtracker import CentroidTracker
from centroidtracker import calculate_slope,calculate_offset, isUnderLine,image_bin

######################################hough#############################################
def findLinesBlue(frame,blur_frame):
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([110, 50, 50]), np.array([130, 255, 255]))
    res = cv2.bitwise_and(blur_frame, blur_frame, mask=mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    edges = cv2.Canny(gray, 50, 200)
    edges = cv2.Canny(res, 50, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=15, maxLineGap=15)

    k=0
 #   while(k<len(lines)):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    #    k = k + 1
    # Show result
 #   cv2.imshow("detektovane linije", frame)


    coord = (0, 0, 0, 0)

    if len(lines) > 1:
        for x1, y1, x2, y2 in lines[1]:
            coord = x1, y1, x2, y2
    else:
        for x1, y1, x2, y2 in lines[0]:
            coord = x1, y1, x2, y2




    return coord


def findLinesGreen(frame,blur_frame):
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([50, 100, 100]), np.array([70, 255, 255]))
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(blur_frame, blur_frame, mask=mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    edges = cv2.Canny(gray, 50, 200)
    edges = cv2.Canny(res, 50, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=15, maxLineGap=15)

    k = 0
 #   while (k < len(lines)):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
       # k = k + 1
    # Show result
    cv2.imshow("detektovane linije", frame)

    coord = (0, 0, 0, 0)

    if len(lines) > 1:
        for x1, y1, x2, y2 in lines[1]:
            coord = x1, y1, x2, y2
    else:
        for x1, y1, x2, y2 in lines[0]:
            coord = x1, y1, x2, y2


    return coord

##############################################################3


def select_roi(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    contours_for_numbers = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)

        if area > 10 and h < 40 and h > 15 and w > 1 and w < 50:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            contours_for_numbers.append(contour)
            region = image_bin[y - 4:y - 4 + h + 8, x - 8:x - 8 + w + 16]
            #mora provera ukoliko se ne ucita dobro slika (assertation !ssize.empty())
            if region.shape[1] == 0:
                continue
            if region.shape[0] == 0:
                continue
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    sorted_regions = sorted_regions = [region[0] for region in regions_array]

    cv2.imshow('ROI', image_orig)

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions, contours_for_numbers

def create_ann():
    '''Implementacija veštačke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)

    return ann

#Korak 5
#skaliranje elemenata
def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    try:
        return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    except Exception as e:
        print(region.shape)
        print(str(e))


#transformacija slike u vektor
def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()


#Priprema za neuronsku mrezu
def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann


#Konverzija
def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)



def neuralN():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_s = []
    for x in x_train[:10:]:
        ret, frame = cv2.threshold(x, 180, 255, cv2.THRESH_BINARY)
        x_train_s.append(frame)

    ann = create_ann()
    ann = train_ann(ann, np.array(prepare_for_ann(x_train_s), np.float32), convert_output(y_train[:10:]))
    model_json = ann.to_json()

    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)

    ann.save_weights("models/model.h5")
    print("Saved model to disk")

    return ann

def check_underline_blue( x, y, w, h, rects, box):
    res = False
    X = int((x + x + w) / 2.0)
    Y = int((y + y + h) / 2.0)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if Y < blue_points[1] + 5 and X > blue_points[0] - 5 and X < blue_points[2] + 5:
        slope = calculate_slope(blue_points[0], blue_points[1], blue_points[2], blue_points[3])
        offset = calculate_offset(slope, blue_points[0], blue_points[1])
        if isUnderLine(x, y, slope, offset):
            rects.append(box)
            res = True
    return res

def check_underline_green( x, y, w, h, rects, box):
    res = False
    X = int((x + x + w) / 2.0)
    Y = int((y + y + h) / 2.0)
    if Y < green_points[1] + 5 and X > green_points[0] - 5 and X < green_points[2] + 5:
        # nadji jednacinu prave i proveri da li je ispod linije
        slope = calculate_slope(green_points[0], green_points[1], green_points[2], green_points[3])
        offset = calculate_offset(slope, green_points[0], green_points[1])
        if isUnderLine(x, y, slope, offset):
            rects.append(box)
            res = True
    return res

def cross_blue(contours, frame):
    rects = []
    k = 0
    while(k<len(contours)):
#    for contour in contours:
        x, y, w, h = cv2.boundingRect(contours[k])
        box = cv2.boundingRect(contours[k])
        box = (x, y, x+w, y+h)
        crossLineBlue = check_underline_blue( x, y, w, h, rects, box)

        if not crossLineBlue:
            print('Nije presao plavu liniju')

        k = k + 1

    objects_blue = ct_blue.update(rects,blue_points[0],blue_points[1],blue_points[2],blue_points[3], frame, loaded_model)


def cross_green(contours, frame):
    rects = []
    k = 0
    while (k < len(contours)):
#    for contour in contours:
        x, y, w, h = cv2.boundingRect(contours[k])
        box = cv2.boundingRect(contours[k])
        box = (x, y, x + w, y + h)
        crossLineGreen = check_underline_green( x, y, w, h, rects, box)

        if not crossLineGreen:
            print('Nije presao zelenu liniju')

        k = k + 1

    objects_green = ct_green.update(rects, green_points[0], green_points[1], green_points[2], green_points[3], frame, loaded_model)



if os.path.isfile("models/model.h5"):

    print("POSTOJI FAJL!")
    fileExists=True
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model.h5")
    print("UCITAN MODEL")
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')
else:
    print("NE POSTOJI FAJL!")
    fileExists=False
    neuralN()



fileOut = open('resources/out.txt', 'w+')
fileOut.write("RA173/2015 Teodora Alempic" + '\n' + "file" + '\t' + '\t' + "sum" + '\n')
fileOut.close()
for i in range(0, 10):

    ct_blue = CentroidTracker()
    ct_blue.setType("BLUE")
    ct_green = CentroidTracker()
    ct_green.setType("GREEN")

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('resources/video-' + str(i) + '.avi')
    ret_begin, frame_begin = cap.read()
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    blur_frame = cv2.GaussianBlur(frame_begin,(5,5),0)

    green_points = findLinesGreen(frame_begin,blur_frame)
    print(green_points)
    blue_points = findLinesBlue(frame_begin,blur_frame)
    print(blue_points)

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret != True:
            break

        # Sklanjamo tackice sa videa
        photo1 = cv2.erode(frame, np.ones((3, 3), np.uint8))

        # pojeli smo brojeve , pa da ih vratimo
        photo2 = cv2.dilate(photo1, np.ones((3, 3), np.uint8))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bin = image_bin(gray)

        selected_regions, numbers, contours = select_roi(frame.copy(), bin)

        cv2.imshow("Frame " + str(i), frame)


        cross_blue(contours, frame)
        cross_green(contours, frame)

        cv2.waitKey(3)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    suma = ct_blue.getSum() - ct_green.getSum()
    f = open('resources/out.txt','a')
    f.write("video-"+ str(i) +".avi " + str(suma) + "\n")
    f.close()
    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()



