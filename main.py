####################################### IMPORT #################################
import json
import sys
from typing import Annotated

from fastapi import FastAPI, UploadFile, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from app import run_inference, to_json, get_images_from_bytes

class Response(BaseModel):
    status: int 
    message: str
    data: dict

###################### FastAPI Setup #############################

# title
app = FastAPI()

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def save_openapi_json():
    '''This function is used to save the OpenAPI documentation 
    data of the FastAPI application to a JSON file. 
    The purpose of saving the OpenAPI documentation data is to have 
    a permanent and offline record of the API specification, 
    which can be used for documentation purposes or 
    to generate client libraries. It is not necessarily needed, 
    but can be helpful in certain scenarios.'''
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    '''
    It basically sends a GET request to the route & hopes to get a "200"
    response code. 
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'OK!'
    }
    '''
    return {'healthcheck': 'OK!'}


######################### MAIN Func #################################

@app.post("/inference")
def inference(images: list[UploadFile]) -> Response:
    """
    Object Detection from an image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections.
    """

    # Convert the image file to an image object
    input_images = get_images_from_bytes([image.file for image in images])

    # Predict from model
    results = run_inference(input_images)
    # json_results: List[str] = to_json(results)
    return Response(
        status=200,
         message="Success!", 
         data={ "results": results }
        )