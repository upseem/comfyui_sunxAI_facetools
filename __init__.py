from .nodes import *
from .InstantID import *




NODE_CLASS_MAPPINGS = {
    'DetectFaces': DetectFaces,
    'DetectFaceByIndex': DetectFaceByIndex,
    'CropFaces': CropFaces,
    'WarpFacesBack': WarpFaceBack,
    "SelectFloatByBool": SelectFloatByBool,


    "InstantIDModelLoader": InstantIDModelLoader,
    "InstantIDFaceAnalysis": InstantIDFaceAnalysis,
    "ApplyInstantID": ApplyInstantID,
    "SaveFaceEmbeds": SaveFaceEmbeds,
    "LoadFaceEmbeds": LoadFaceEmbeds,

    "VAEDecodeNew": VAEDecodeNew,
    "VAEEncodeNew": VAEEncodeNew,

    'ColorAdjustNew(FaceParsing)': ColorAdjust,

    "SaveImageWebsocketNew": SaveImageWebsocket,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'DetectFaces': 'DetectFaces',
    'DetectFaceByIndex': 'Detect Face By Index',
    'CropFaces': 'CropFaces',
    'WarpFacesBack': 'Warp Faces Back',
    "SelectFloatByBool": "Select Float (Bool)",

    "InstantIDModelLoader": "Load InstantID Model",
    "InstantIDFaceAnalysis": "InstantID Face Analysis",
    "ApplyInstantID": "Apply InstantID",
    "SaveFaceEmbeds": "Save Face Embeds",
    "LoadFaceEmbeds": "Load Face Embeds",

    "VAEDecodeNew": "VAE Decode New",
    "VAEEncodeNew": "VAE Encode New",

    'ColorAdjustNew(FaceParsing)': 'Color Adjust (Face Parsing) New',
    "SaveImageWebsocketNew": "Save Image Websocket New To JPG",
}




__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
