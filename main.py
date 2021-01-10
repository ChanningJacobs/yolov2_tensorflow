import sys
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.insert(1, '/media/channing/New Volume/Data/cocoapi/PythonAPI/')
from pycocotools.coco import COCO
from collections import defaultdict

#######################################
# CREATE TF DATASET USING PYCOCOTOOLS #
#######################################

coco_img_path = '/media/channing/New Volume/Data/coco/images/'
coco_ann_path = '/media/channing/New Volume/Data/coco/annotations/'

train_img_path = coco_img_path + 'train2017/'
train_ann_path = coco_ann_path + 'instances_train2017.json'
val_img_path = coco_img_path + 'val2017/'
val_ann_path = coco_ann_path + 'instances_val2017.json'
test_path = coco_img_path + 'test2017/'

train_coco = COCO(train_ann_path)
val_coco = COCO(val_ann_path)


def coco_id_pad(img_id):
    ''' Convert img_id to str, pad length to 12 with '0's and add '.jpg'.'''
    return str(img_id).zfill(12) + '.jpg'


print('Creating training and validation labels.')
train_dict = defaultdict(lambda: [0]*90)
val_dict = defaultdict(lambda: [0]*90)
for ann in train_coco.loadAnns(train_coco.getAnnIds()):
    path = train_img_path + coco_id_pad(ann['image_id'])
    train_dict[path][ann['category_id'] - 1] = 1
for ann in val_coco.loadAnns(val_coco.getAnnIds()):
    path = val_img_path + coco_id_pad(ann['image_id'])
    val_dict[path][ann['category_id'] - 1] = 1
# Create a tuple of 0D string tensors and 1D multihot tensors
train_labels = (list(train_dict.keys()), list(train_dict.values()))
val_labels = (list(val_dict.keys()), list(val_dict.values()))


def load_image(*example):
    img = tf.io.read_file(example[0])
    img = tf.image.decode_image(img, channels=3, dtype=tf.dtypes.float32)
    print(tf.math.reduce_max(img))
    return (tf.image.resize_with_crop_or_pad(img, 512, 512), example[1])


print('Done prepping labels. Reading images to create dataset.')
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices(train_labels).shuffle(len(train_labels)).map(load_image, AUTOTUNE).batch(32).prefetch(AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices(val_labels).shuffle(len(val_labels)).map(load_image, AUTOTUNE).batch(32).prefetch(AUTOTUNE)
print('Done creating dataset. Starting Welford running mean/std.')


def view_ds_batch(tf_ds):
    for batch in tf_ds:
        for i in range(len(batch[1])):
            print(f'Label:{batch[1][i]}')
            plt.imshow(batch[0][i])
            plt.show()
            plt.close()
        break


def welford_running_mean_std(aggregate, new_val):
    count, mean, m2 = aggregate
    count += 1
    delta1 = new_val - mean
    mean.assign_add(delta1 / count)
    delta2 = new_val - mean
    m2.assign_add(delta1 * delta2)
    return (count, mean, m2)
'''
iii = 0
for batch in train_ds:
    iii += 1
    print(iii)
    #for image in batch[0]:
exit()
ii = 0
aggregate = (0, tf.Variable(tf.zeros([512, 512, 3])), tf.Variable(tf.zeros([512, 512, 3])))
for batch in train_ds:
    ii += 1
    for image in batch[0]:
        aggregate = welford_running_mean_std(aggregate, image)
        print(aggregate)
    if ii == 2:
        break
    #print(aggregate[0])
    #print(aggregate[1].shape)
    #print(aggregate[2].shape)
print(aggregate)

exit()
#################################################
# CREATE AND TRAIN AN IMAGE CLASSIFIER FOR COCO #
#################################################
exit()
'''
class Classifier(tf.keras.Model):

    def __init__(self, mean, std):
        super(Classifier, self).__init__()
        self.mean = mean
        self.std = std
        # Training only data augmentation layers
        # TODO rescale should stay...
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)
        self.rand_contrast = tf.keras.layers.experimental.preprocessing.RandomContrast(0.9)
        self.rand_flip = tf.keras.layers.experimental.preprocessing.RandomFlip()
        self.rand_rotation = tf.keras.layers.experimental.preprocessing.RandomRotation(0.8)
        # Normalization layer based on training data (weights saved)
        #self.normalize = tf.keras.layers.experimental.preprocessing.Normalization(mean=self.mean, variance=self.std**2)

    def call(self, x):
        #x = self.normalize(x)
        # Data augmentation
        #x = self.rescale(x)
        x = self.rand_contrast(x)
        x = self.rand_flip(x)
        x = self.rand_rotation(x)
        return x


model = Classifier(0, 0)

for batch in train_ds:
    result = model(batch[0])
    print(result.shape)
    for i in range(len(result)):
        print(f'Label:{batch[1][i]}')
        image = result[i]
        print(tf.math.reduce_min(batch[0][i]))
        print(tf.math.reduce_max(batch[0][i]))
        print(tf.math.reduce_max(image))
        print(tf.math.reduce_min(image))
        #print(min(load_image))
        #print(max(image))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(batch[0][i])
        ax[1].imshow(result[i])
        plt.show()
# view_ds_batch(train_ds)
# view_ds_batch(val_ds)

#import time
#t = time.time()
#for x in train_ds:
#    x.numpy()
#print(time.time() - t)

# Preprocessing
# Utilities
    # IOU
    # NMS
        # Discard low probability of object boxes
        # Each output prediction sorted into a per class list
        # Pick box with highest Pc and discard remaining with high IOU
    # AB
        # kmeans
