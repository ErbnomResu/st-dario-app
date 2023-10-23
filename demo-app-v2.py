import cv2, io, os
import streamlit as st
import numpy as np
from zipfile import ZipFile
from PIL import Image


class ImageFile:
    def __init__(self, name, image):
        self.name = name
        self.image_extension = ".jpg"
        self.image = self.check_channels(image)
        self.image_processed = None 
        self.image_bytes = None

    def get_name(self):
        return self.name

    def get_image(self):
        return self.image


    def get_processed_image(self):
        return self.processed

    # Conversion from 4 channels to 3 channels
    def check_channels(self, new_image):
        if(len(new_image.shape)) > 2 and new_image.shape[2] == 4:
            return cv2.cvtColor(new_image, cv2.COLOR_BGRA2BGR)

        return new_image


    def get_image_bytes(self):
        return self.image_bytes
        

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


def data_to_buffer(image, extension="JPEG"):
    buffer = io.BytesIO()
    img_to_save = Image.fromarray(image)
    img_to_save.save(buffer, format=extension)

    return buffer





def main_loop():
	st.title("EL TRABAJITO")

	st.text("Instrucciones: primero arrastrar las imagenes a la zona inferior y a continuación pulsar el botón de descarga del zip.")
	st.text("Cualquier detalle consulte con mi secretario Bertin.")
	st.text("Que disfrutes!")

	# read file uploaded and reset uploader field
	files_list=None

	with st.form("my_form", clear_on_submit=True):
		files_list = st.file_uploader("Upload your images/s", type = ["jpg", "png", "jpeg"], accept_multiple_files=True)
		

		zip_name = st.text_input("Nombre de la carpeta a descargar: ", "default.zip")

		st.text("Pulsa para empezar o presiona Enter (en el nombre):")

		submitted=st.form_submit_button("Hit me, daddy!")

	if not files_list:
		return
	
	if  not submitted:
		return
	
	# if user hits submit
	st.text("___________________________________________________")

	st.text(f"Comenzando el procesamiento de {len(files_list)} imagenes...")
	# Iniciando image procesor
	img_processor = ProcessImageFile()

	counter = st.text(f"Procesados 0 / {len(files_list)}")
	with ZipFile("output.zip", "w") as zip_object:
		for index, file in enumerate(files_list):
			clean_name = file.name.split(".")[0]

			#st.text(f"Procesando imagen: {clean_name}")
			counter.text(f"Procesados => {index + 1} / {len(files_list)}")
			# load file as image and convert it to array
			file_img = Image.open(file)
			file_img = np.array(file_img)

			# create image file object and processed it
			img_obj = ImageFile(clean_name, file_img)
			img_obj.image_processed = img_processor.process_image(img_obj)

			# resize half the size
			#img_obj.image_processed = cv2.resize(img_obj.image_processed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

			# Create Image Bytes Object
			img_obj.image_bytes = data_to_buffer(img_obj.image_processed).getvalue() 

			
			zip_object.writestr(img_obj.get_name() + img_obj.image_extension, img_obj.get_image_bytes())

			

		st.text("Finalizando procesamiento de imagenes...")

		st.text("___________________________________________________")

		

		if(zip_name != "" or zip_name != " "):
			split_name = zip_name.split(".")
			zip_name = split_name[0] + ".zip"
		else:
			zip_name = "default.zip"

		#zip_name = "output.zip"

		st.text(f"{zip_name}")

		st.text("Pulsa el siguiente botón para descargar el zip con los archivos procesados")
		with open("output.zip", "rb") as zip_object:
			
			btn = st.download_button (
				label="Download zip",
				data=zip_object,
				file_name=zip_name,
				mime="application/zip"
			)

			if(btn is not None):
				files_list = None
				submitted = None

		

		st.text("___________________________________________________")
		st.text("Muestra de la labor realizada...")
		st.image([img_obj.get_image(), img_obj.image_processed])  

		




if __name__ == "__main__":
    main_loop()