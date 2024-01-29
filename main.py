import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
# from streamlit_dimensions import st_dimensions
import cv2 as cv
import numpy as np
import base64
import requests
import os

from txt2img import gen_img


# have to add text input in streamlit
# inp = input("what image you want to generate: ")

gen_img_api = np.ones((500, 500, 3))*255

# api endpoint for background removal
api_endpoint_br = "https://rituramojha.ap-south-1.modelbit.com/v1/remove_background/latest"

# set layout
st.set_page_config(layout='wide')
st.title("Image Background Replacer")

col1, col2 = st.columns(2)


# -----------get the width of the col2
# with col2:
#     screen_dim = st_dimensions(key='col1') #got width=680
#     st.write(screen_dim['width'])


# file uploader
file = col2.file_uploader('Upload Image', type=['jpeg', 'jpg', 'png'])


# read image
if file is not None:
    image = Image.open(file).convert('RGB')
    # the '680' is obtained from the width of columns earlier
    # image = image.resize((680, int(
    #     image.height*680*0.5/(image.width))))
    image = cv.resize(np.asarray(image), (680, int(
        image.height*680/(image.width))), interpolation=cv.INTER_AREA)

    # create buttons
    col2_1, col2_2 = col2.columns(2)

    # getting the coordinates of the image where the user clicked
    placeholder0 = col1.empty()

    with placeholder0:
        value = im_coordinates(image, key='initial')

    if col2_1.button('Original', use_container_width=True):
        placeholder0.empty()
        placeholder1 = col1.empty()
        with placeholder1:
            placeholder1.image(image)
            # rerun from the top to enable user to click at another place in the image
            st.rerun()

    if col2_2.button('Replace Background', type='primary', use_container_width=True):
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
            response = requests.post(api_endpoint_br, json=api_dataa)

            # getting the output from the API
            final_image = response.json()['data']

            # decoding the received encoded image from the function into simple numpy array
            final_image_bytes = base64.b64decode(final_image)
            final_image_ap = cv.imdecode(np.frombuffer(
                final_image_bytes, dtype=np.uint8), cv.IMREAD_UNCHANGED)

            # replacing work from here
            bg_img = cv.resize(
                gen_img_api, (final_image_ap.shape[1], final_image_ap.shape[0]))
            x = final_image_ap[:, :, 3]
            y = cv.bitwise_not(x)
            n = cv.merge((bg_img, y))

            # Create a mask for pixels where the alpha channel of n is 0
            alpha_zero_mask = n[:, :, 3] == 0

            # Use the mask to update the values in n with the corresponding values from final_image_api
            n[alpha_zero_mask] = final_image_ap[alpha_zero_mask]

            # cv.imwrite(filename, final_image_ap)
            cv.imwrite(filename, n)

        with placeholder2:
            placeholder2.image(n, use_column_width=True)
