# Klasa CentroidTracker je uzeta sa sajta (u posteru objasnjenje sta ona tacno radi):
# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/


# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import matplotlib


import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def calculate_slope(x1,y1,x2,y2):
    k = (y2 - y1) / (x2 - x1)
    return k;

def calculate_offset(k,x1,y1):
    n = -k*x1 + y1
    return n

def isUnderLine(x1,y1,k,n):
    temp = k*x1+n
    if temp <= y1:
        return True
    else:
        return False

def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255 - image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def calculate_distance(x1, y1, x2, y2, x0, y0):
    denominator = (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1

    if denominator > 0:
        denominator = denominator * (-1)
    else:
        denominator = denominator * (-1)

    numerator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return denominator / numerator

#NEURONSKE MREZE

# def create_ann():
#     model = Sequential()
#     model.add(Dense(512, input_shape=(784,)))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(10))
#     model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam')
#
# def train_ann(ann):
#     '''Obucavanje vestacke neuronske mreze'''
#     # spremi trening skup
#     nb_classes = 10
#     # ucitaj trening skup pomocu keras biblioteke
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
#     # pretvori slike iz 28x28 u vektor od 784 elemenata
#     X_train = X_train.reshape(60000, 784)
#     X_train = X_train.astype('float32')
#
#     # skalirati elemente od 0 - 255 na 0 - 1
#     X_train /= 255
#
#     # one hot prezentacija za svaki broj
#     Y_train = np_utils.to_categorical(y_train, nb_classes)
#
#     # obucavanje neuronske mreze
#     ann.fit(X_train, Y_train, batch_size=256, epochs=500, verbose=1)
#
#     return ann


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    try:
        return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    except Exception as e:
        print(region.shape)
        print(str(e))


#Korak 5
#skaliranje elemenata
def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255


#transformacija slike u vektor
def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()

def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def result(boxes, frame, regions):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    img = image_bin(image_gray(blur))
    region = img[boxes[1] - 4:boxes[1] - 4 + boxes[3] - boxes[1] + 8,
             boxes[0] - 8:boxes[0] - 8 + boxes[2] - boxes[0] + 16]
    resized = resize_region(region)
    scaled = scale_to_range(resized)
    vector = matrix_to_vector(scaled)
    regions.append(vector)
    return regions

class CentroidTracker():
    def __init__(self, maxDisappeared=2):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.added = OrderedDict()
        self.type = ""
        self.total_sum = 0

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
    def getAdded(self):
        return self.added

    def setType(self,trackerType):
        self.type = trackerType
    def getSum(self):
        return self.total_sum



    # u tracker se objekat registruje tek kada prodje ispod linije
    # cim se registruje vrsi se dodavanje/oduzimanje u zavisnosti od vrste trackere i vrsi se dalje pracenje objekta
    def register(self, centroid, boxes, frame, loaded_model):
        regions = []
        alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        regions1 = result(boxes,frame,regions)
        result_ann = loaded_model.predict(np.array(regions1, np.float32))
        if not (self.type == "BLUE"):
            temp = display_result(result_ann, alphabet)
            print(temp)
            self.total_sum += temp[0]
        else:
            temp = display_result(result_ann, alphabet)
            print(temp)
            self.total_sum += temp[0]
        # when registering an object we use the next available object
        # ID to store the centroid
        # ako je registrovan to znaci da je prosao ipsod linije - dodat je
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def getAdded(self):
        return self.added

    def update(self, rects,x1,y1,x2,y2, frame, loaded_model):
        # proveri da li su se javile konture
        if len(rects) == 0:
            remove = []
            # proveri sve koje vec pratis i vidi da li su ispali iz opsega
            # ili su prekoracili broj frejmova za koje su "nestali"
            for objectID in self.objects.keys():
                centroid = self.objects[objectID]
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    remove.append(objectID)

                forDelete(remove,centroid,objectID,y1)


            for r in remove:
                self.deregister(r)
            # izadji odma posto nema novih kontura
            return self.objects

        # inicijalizuj input centroide za trenutni frejm
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        boxes = []
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            boxes.append(rects[i])

        # ako ne pratimo objekte uzmi input centroida i registruj ih
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], boxes[i], frame, loaded_model)

        # u suprotnom, pratimo vec objekte i treba da ih povezemo sa
        # odgovarajucim centroidima iz inputa
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # izracunaj daljinu izmedju objekata koje pratis i novih pozicija pojedinacno - input centroida
            # koji predstavljaju pomeraje u odnosu na proslu poziciju
            # D je niz nizova, svaki niz sadrzi daljine izmedju objekta koji se prati i novih pozicija
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # D.min(axis=1) vraca minimalno rastojanje izmedju objekta koji se prati i novih pozicija
            # sortiraj kako bi dobio id-eve
            rows = D.min(axis=1).argsort()

            # cols sadrzi informaciju o tome na kom indeksu se nalazi nova pozicija
            # za svaki objekat iz input centroida
            # uradi se argmin() za odgovarajuce id-eve (indexe) iz rows
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                # row je ID a col je indeks pozizicije nove vrednosti iz inputa
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # ako je broj objects veci od broja inputa mora se proveriti da li je
            # object nestao - moze se desiti ako se preklope brojevi ili ako prelazi preko linije
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                remove = []
                for row in unusedRows:
                    objectID = objectIDs[row]
                    centroid = self.objects[objectID]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

                    forDelete(remove, centroid, objectID,y1)

                    for r in remove:
                        self.deregister(r)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], boxes[col], frame, loaded_model)

        # return the set of trackable objects
        return self.objects

def forDelete(remove,centroid, objectID, y1):
    if centroid[1] > y1 + 5 and centroid[0] < x1 - 5 and centroid[0] > x2 + 5:
        if objectID not in remove:
            remove.append(objectID)

