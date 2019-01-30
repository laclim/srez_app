import tensorflow as tf
import os
import scipy.misc

filenames="C:\\Users\\USER\\AppData\\Local\\Programs\\Python\\Python36\\GAN super resolution\\srez-master\\dataset\\00001.jpg"
# filename_queue = tf.train.string_input_producer(filenames)
# key, value = reader.read(filename_queue)
channels = 3
image = tf.image.decode_jpeg(filenames, channels=channels, name="dataset_image")
print(image)

filename = 'sample1.png' 
filename = os.path.join("C:\\Users\\USER\\AppData\\Local\\Programs\\Python\\Python36\\GAN super resolution\\srez-master\\", filename)
scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
print("    Saved %s" % (filename,))