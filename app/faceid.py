# Import kivy dependecies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Kivy uiux components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Other kivy components
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


class CamApp(App):

    def build(self):
        self.webcam = Image(size_hint=(1, .8))
        self.button = Button(
            text='Verify', on_press=self.verify, size_hint=(1, .1)
        )
        self.verification_label = Label(
            text='Verification Uninitiated', size_hint=(1, .1)
        )

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.webcam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load model
        self.model = tf.keras.models.load_model(
            'siamesemodel.h5', custom_objects={'L1Dist': L1Dist}
        )

        # Video capture
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    # Run continuously to get webcam feed
    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 650:650+250, :]
        frame = cv2.resize(frame, (1000, 1000))

        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr'
        )
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = img_texture

    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image
        img = tf.io.decode_jpeg(byte_img)

        # Preprocessing - resizing the image to be 105x105x3
        img = tf.image.resize(img, (105, 105))
        # Scale image to between 0 and 1
        img = img/255.0
        return img

    def verify(self, *args):
        """
        model: siamese model
        detection_threshold: metric above which a prediction is classified as positive
        verification_threshold: proportion of positive predictions / total positive samples
        """

        detection_threshold = 0.5
        verification_threshold = 0.6

        # Capture input images
        INPUT_IMG_PATH = os.path.join(
            'application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 650:650+250, :]
        cv2.imwrite(INPUT_IMG_PATH, frame)

        results = []
        verification_images = os.listdir(os.path.join(
            'application_data', 'verification_images'))
        for image in verification_images:
            input_img = self.preprocess(os.path.join(
                'application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join(
                'application_data', 'verification_images', image))

            result = self.model.predict(
                list(np.expand_dims([input_img, validation_img], axis=1)),
                verbose=0
            )
            results.append(result)
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection/len(verification_images)
        verified = verification > verification_threshold

        # Set verification text
        self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Log details
        Logger.info(results)
        Logger.info(verification)
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.5))
        Logger.info(np.sum(np.array(results) > 0.8))

        return results, verified


if __name__ == '__main__':
    CamApp().run()
