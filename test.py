import zipfile
import tensorflow as tf
file_path = "../data/text8.zip"
with zipfile.ZipFile(file_path) as f:
    words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    print(tf.compat.as_str(f.read(f.namelist()[0])))
