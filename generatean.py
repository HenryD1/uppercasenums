import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display


#Dataset, we will find and clean later
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data() #Load data from preset mnist handwriting
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]. Before they were color values in between 0 and 256

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE) #shuffle dataset, will be annoying later


def make_generator_model(): #were gonna alter this later to have a lowercase letter input. So Ill have to find where randomness xomes from.
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,))) #for now I'll operate under the assumption that units = 7*7*256, which implies a 49*256-dimensional output space
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

      #We may have to change conv 2d to conv3d given that we're presenting the data as 3-dimensional.

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100]) #seed will be replaced by correct lowercase; we want to replace noise with an input lowercaseletter
generated_image = generator(noise, training=False) #change this to generated_uppercaseletter #what does training parameter do. #We're gonna run it on input of lowercaseletter, not randomness. Ok this parameter just means that it doesnt run the training loop which it expects to do due to tf, so we can see an example of what an initalized nn does to a random seed

plt.imshow(generated_image[0, :, :, 0], cmap='gray') #this should be unchanged.



def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1])) #We should have it take as input the real lowercase (ie the input the generator gets) and the fake uppercase. Another 28x28x2
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3)) #i dont just dropout params, we could test this with many

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) #this is still a decision 

    return model


discriminator = make_discriminator_model()
decision = discriminator(generated_image) 
print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) #sure, lets use this.

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) #we want real_outpit to be 1. Because the discriminator ought to output 1 when it sees a real image.
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

#Define the training loop.
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim]) #there will be no seed.

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# ok honestly no idea what this does but I will just be as delicate as possible so as to not break it.
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim]) #why is noise redefined here? I suspect it has to do with how variables work in python

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: #again idk what "with" does but I will make substitutions
    #ok I think it instantiates an object?
    #Im just gonna not touch it and hope it doesn't break.
      #1. first generate images
      generated_images = generator(noise, training=True) #we'll change this to generated_upper = generator(lower, training = True)
      #2. Then discriminate those images
      real_output = discriminator(images, training=True) #upper (cleaned data)
      fake_output = discriminator(generated_images, training=True) #change to generated_upper
      #compute both losses (which needs both to run)
      gen_loss = generator_loss(fake_output) #just compute generator loss as defined aboe, no need to change anything
      disc_loss = discriminator_loss(real_output, fake_output) #-

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables) #gen.tape is an object of sorts that can compute the gradient
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables) #we use disc.tape so they don't get confused

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)) # this just seems like more necessary tf code
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables)) #this is more necessary tf code


#Training
def train(dataset, epochs): #we will run this on a batched and shuffled lower/uppercase model
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset: #dataset now will be cleaned batched and shuffled lowercase/uppercase
      train_step(image_batch) #I'll figure out exactly how the image batches are presented so the train_step API can read them

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True) #what does this do? I think probably just runs it one more time
  generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input): #how can we declare this after it is called?
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

#--- up to training.
# we will call this to train the generator and discriminator simultaneously.
#We must ensure that they train at a similar rate

#At the beginning of the training, the training images should look like 1 layer of 

train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#Create a gif
# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
  display_image(EPOCHS)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
  display.Image(filename=anim_file)

try:
    from google.colab import files
except ImportError:
   pass
else:
  files.download(anim_file)

  #This all might need to be changed to ignore a first layer.


