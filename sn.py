import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score


from tensorflow.keras.datasets import mnist
from tensorflow.keras import regularizers


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

x_train, x_test = x_train / 255-0.5, x_test / 255-0.5


y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)
x_train=np.expand_dims(x_train,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, shear_range=10,
                        height_shift_range=0.1, zoom_range=0.2)
datagen.fit(x_train)
datagen2 = tf.keras.preprocessing.image.ImageDataGenerator()
datagen2.fit(x_test)


def gen_layer(layer, prev, filters=128, k_size=[3,3], pool=False, dropout=False):
        bn_decay = 0.95

        conv_dropout = 0.2 
        WEIGHT_DECAY = 0.005 

        conv = tf.keras.layers.Conv2D(
                filters, 
                kernel_size=k_size, 
                strides=[1, 1], 
                padding="SAME", 
                kernel_initializer=tf.keras.initializers.GlorotUniform(), 
                bias_initializer=tf.constant_initializer(0.1), 
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY), 
                name=f'conv{layer}')(prev)

        bn = tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=bn_decay)(conv)
        if pool:
            pool = tf.nn.max_pool(bn, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")    
            act = tf.keras.layers.LeakyReLU(alpha=0.1, name=f'act{layer}')(pool)
        else:
            act = tf.keras.layers.LeakyReLU(alpha=0.1, name=f'act{layer}')(bn)

        if not dropout:
            return act
        drop = tf.keras.layers.Dropout(conv_dropout)(act, training=True)
        
        return drop


def gen_model():
        WEIGHT_DECAY = 0.005 

        s = tf.keras.Input(shape=x_train.shape[1:]) 
        
        l1 = gen_layer(1, s, filters=64)
        l2 = gen_layer(2, l1, pool=True)
        l3 = gen_layer(3, l2, pool=True)
        l4 = gen_layer(4, l3, pool=True)
        l5 = gen_layer(5, l4)
        l6 = gen_layer(6, l5)
        l7 = gen_layer(7, l6, pool=True)
        l8 = gen_layer(8, l7, pool=True)
        l9 = gen_layer(9, l8, pool=True)
        l10 = gen_layer(10, l9)
        l11 = gen_layer(11, l10, k_size=[1,1])
        l12 = gen_layer(12, l11, k_size=[1,1], pool=True)

        conv13 = tf.keras.layers.Conv2D( 
                128, 
                kernel_size=[3, 3], 
                strides=[1, 1],
                padding="SAME",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                bias_initializer=tf.constant_initializer(0.1),
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                name='conv13')(l12)

        pool13 = tf.nn.max_pool(conv13, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
        avg13 = tf.keras.layers.GlobalAveragePooling2D(name='avg13')(pool13)

        x = tf.keras.layers.Dense(10,activation='softmax',use_bias=False,
                                  kernel_regularizer=tf.keras.regularizers.l1(0.00025))(avg13) # this make stacking better
        return tf.keras.Model(inputs=s, outputs=x)


batch_size=32
supermodel=[]
model=gen_model()                
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size,shuffle=True),
            steps_per_epoch=len(x_train) / batch_size, epochs=13,verbose=0)
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size,shuffle=True),
            steps_per_epoch=len(x_train) / batch_size, epochs=3,verbose=0)
model.compile(optimizer=optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size,shuffle=True),
            steps_per_epoch=len(x_train) / batch_size, epochs=3,verbose=0)
model.fit(x_train, y_train, batch_size=batch_size,shuffle=True, epochs=1,verbose=0)
print('acc:',accuracy_score(np.argmax(y_test,axis=1),np.argmax(model.predict(x_test),axis=1)))


model.save("sn")

