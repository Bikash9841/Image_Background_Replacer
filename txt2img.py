import requests
import numpy as np
import cv2 as cv
import base64

api_endpoint_ig = "https://rituramojha.ap-south-1.modelbit.com/v1/generate_image/latest"


def gen_img(prompt):
    '''
    takes in prompt and return ndarray image.
    '''
    # prompt = "photo realistic murder scene with vibrant lights"

    # calling from the api
    api_data = {"data": [prompt]}
    response = requests.post(api_endpoint_ig, json=api_data)

    # print(response.json())
    final_image = response.json()['data']

    # decoding the received encoded image from the function into simple numpy array
    final_image_bytes = base64.b64decode(final_image)
    gen_img_api = cv.imdecode(np.frombuffer(
        final_image_bytes, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    return gen_img_api
