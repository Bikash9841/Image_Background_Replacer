import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
# from streamlit_dimensions import st_dimensions
import cv2 as cv
import numpy as np
import base64
import requests
import os


api_endpoint = "https://rituramojha.ap-south-1.modelbit.com/v1/remove_background/latest"


# set layout
st.set_page_config(layout='wide')

col1, col2 = st.columns(2)

# -----------get the width of the col2
# with col2:
#     screen_dim = st_dimensions(key='col1') #got width=680
#     st.write(screen_dim['width'])

# --------to visualize the columns
# # Define the width and height of the rectangle
# width = screen_dim['width']
# height = 100

# Create a simple rectangle using HTML and CSS
# rectangle_style = f"width: {width}px; height: {height}px; background-color: lightblue; border: 1px solid black;"
# html_code = f'<div style="{rectangle_style}"></div>'

# Display the rectangle using st.markdown
# st.markdown(html_code, unsafe_allow_html=True)

# file uploader
file = col2.file_uploader('Upload Image', type=['jpeg', 'jpg', 'png'])

# read image
if file is not None:
    image = Image.open(file).convert('RGB')
    # the '680' is obtained from the width of columns earlier
    image = image.resize((680, int(
        image.height*680/(image.width))))

    # create buttons
    col2_1, col2_2 = col2.columns(2)

    # getting the coordinates of the image where the user clicked
    placeholder0 = col1.empty()
    with placeholder0:
        value = im_coordinates(image, key='initial')
    # if value is not None:
        # st.write(value)

    if col2_1.button('Original', use_container_width=True):
        placeholder0.empty()
        placeholder1 = col1.empty()
        with placeholder1:
            # col1.image(image, use_column_width=True)
            value = im_coordinates(image, key='next')

    if col2_2.button('Remove Background', type='primary', use_container_width=True):
        placeholder0.empty()
        placeholder2 = col1.empty()

        # giving filename to the uploaded image and if its exists earlier
        # then no need to call for API evertime user transition from original
        # image to remove background buttons for same image.->makes process faster
        filename = '{}_{}_{}.png'.format('test', value['x'], value['y'])
        if os.path.exists(filename):
            final_image_ap = cv.imread(filename, cv.IMREAD_UNCHANGED)

        else:

            # encoding the image into base64 format to send to the API
            _, img_bytes = cv.imencode('.png', np.asarray(image))
            img_bytes = img_bytes.tobytes()
            image_bytes_encoded = base64.b64encode(img_bytes).decode('utf-8')

            # calling from the api
            api_dataa = {"data": [image_bytes_encoded, value['x'], value['y']]}
            response = requests.post(api_endpoint, json=api_dataa)

            # getting the output from the API
            final_image = response.json()['data']

            # decoding the received encoded image from the function into simple numpy array
            final_image_bytes = base64.b64decode(final_image)
            final_image_ap = cv.imdecode(np.frombuffer(
                final_image_bytes, dtype=np.uint8), cv.IMREAD_UNCHANGED)

            cv.imwrite(filename, cv.cvtColor(final_image_ap, cv.COLOR_BGR2RGB))

        with placeholder2:
            col1.image(final_image_ap, use_column_width=True)
