import cv2
import streamlit as st
import numpy as np
from PIL import Image
import io


def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

class ImageFile:
    def __init__(self, name, image):
        self.name = name
        self.image = self.check_channels(image)
        self.processed = None 

    def get_name(self):
        return self.name

    def get_image(self):
        return self.image


    def get_processed_image(self):
        return self.processed

    def check_channels(self, new_image):
        if(len(new_image.shape)) > 2 and new_image.shape[2] == 4:
            return cv2.cvtColor(new_image, cv2.COLOR_BGRA2BGR)

        return new_image


class ProcessImageFile:
	def __init__(self):
		self.img_format = ""
		self.img_format_size = (0,0,0)
		self.canvas = None
		self.img_canvas_dim = (0,0)

		self.font = cv2.FONT_HERSHEY_DUPLEX
		self.font_scale = 3
		self.font_thickness = 3
		self.font_color = (0,0,0)
		self.font_pos = (0,0)

		self.set_format_px_size("DIN A4")
		self.create_white_canvas() 

	def set_format_px_size(self, new_format = ""):

		if(new_format == ""):
			return

		self.img_format = new_format
		if(new_format == "DIN A0"):
			self.img_format_size = (3508, 2480,3)

		elif(new_format == "DIN A1"):
			self.img_format_size =  (3508, 2480,3)

		elif(new_format == "DIN A2"):
			self.img_format_size = (3508, 2480,3)

		elif(new_format == "DIN A3"):
			self.img_format_size = (3508, 2480,3)

		elif(new_format == "DIN A4"):
			self.img_format_size =  (3508, 2480,3)

		else:
			self.img_format_size = (3508, 2480,3)


	def get_format(self):
		return self.img_format

	def create_white_canvas(self):

		self.canvas = np.ones(self.img_format_size, dtype = np.uint8)
		self.canvas = 255 * self.canvas

		self.set_img_canvas_dim()

	def set_img_canvas_dim(self):
		dim_x = int(self.canvas.shape[1] * 0.9)
		dim_y = 1250

		self.img_canvas_dim = (dim_x, dim_y)

	def change_format(self, new_format= ""):
		self.set_format_px_size(new_format)
		self.create_white_canvas()

 
	def process_image(self, image_file = None):

		if(image_file == None):
			return None

		resized = self.resize_image(image_file.get_image())

		canvas_tmp = self.canvas.copy()

		self.blend_image_in_canvas(canvas_tmp, resized)

		self.set_header_text(canvas_tmp, image_file.get_name())

		return canvas_tmp


	def resize_image(self, image):
		return cv2.resize(image, self.img_canvas_dim, interpolation = cv2.INTER_AREA)


	def blend_image_in_canvas(self, canvas, image):
		# Set Image Position into Canvas
		h_canvas, w_canvas = canvas.shape[:2]
		h_img, w_img = image.shape[:2]

		cy, cx = (h_canvas - h_img) // 2, (w_canvas - w_img) // 2

		canvas[150:150+h_img, cx:cx+w_img] = image


	def set_header_text(self, canvas, text = ""):
		# Get Boundary text size
		text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]

		# get coords base on boundary
		text_x = int((canvas.shape[1] - text_size[0]) / 2)
		text_y = int(canvas.shape[0] * 0.03)

		font_pos = (text_x, text_y)


		cv2.putText(canvas, text, font_pos, self.font, self.font_scale, self.font_color,  self.font_thickness)


def main_loop():
    st.title("EL TRABAJITO")
    st.subheader("Aplicacion destinada a realizar un crop and past de las imagenes de trabajo en un DIN A4")
    st.text("We use OpenCV and Streamlit for this 'trabajito' ")

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None
    
    st.text(f"Nombre del archivo {image_file.name}")


    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    

    img_file = ImageFile(image_file.name, original_image)

    #processed_image = blur_image(original_image, blur_rate)
    #processed_image = brighten_image(processed_image, brightness_amount)

    #if apply_enhancement_filter:
    #    processed_image = enhance_details(processed_image)

    img_processor = ProcessImageFile()
    img_file.processed =  img_processor.process_image(img_file)

    buffer = io.BytesIO()

    img_to_save = Image.fromarray(img_file.processed)

    img_to_save.save(buffer, format="PNG")

    btn = st.download_button(
	    label="Download image",
	    data=buffer,
	    file_name=img_file.get_name(),
	    mime="image/png"
    )



    st.text("Original Image vs Processed Image")
    st.image([original_image, img_file.processed])





if __name__ == '__main__':
    main_loop()