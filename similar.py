import argparse
import os

from packaging import version
import azure.cognitiveservices.vision.face as faceAPI
if version.parse(faceAPI.__version__) <= version.parse('0.3.0'):
    from azure.cognitiveservices.vision.face._face_client import FaceClient  # The main interface to access Azure face API
else:
    from azure.cognitiveservices.vision.face import FaceClient

from msrest.authentication import CognitiveServicesCredentials  # To hold the subscription key

from mlhub.pkg import is_url, get_private

from utils import (
    azface_detect,
    azface_similar,
    get_abspath,
    print_similar_results,
    stop,
)

# ----------------------------------------------------------------------
# Parse command line arguments
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    prog='similar',
    description='Find similar faces between images.'
)

parser.add_argument(
    'target',
    help='path or URL of a photo of the faces to be found')

parser.add_argument(
    'candidate',
    help='path or URL of a photo to find expected target faces')

args = parser.parse_args()

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
target_url = args.target if is_url(args.target) else get_abspath(args.target)  # Get the photo of target faces
candidate_url = args.candidate if is_url(args.candidate) else get_abspath(args.candidate)  # Get the photo to be checked

if os.path.isdir(target_url) or os.path.isdir(candidate_url):
    stop("Only one photo allowed!")

# ----------------------------------------------------------------------
# Prepare Face API client
# ----------------------------------------------------------------------

# Request subscription key and endpoint from user.
subscription_key, endpoint = get_private()

credentials = CognitiveServicesCredentials(subscription_key)  # Set credentials
client = FaceClient(endpoint, credentials)  # Setup Azure face API client


# ----------------------------------------------------------------------
# Detect faces
# ----------------------------------------------------------------------

target_faces = azface_detect(client, target_url)
candidate_faces = azface_detect(client, candidate_url)
if not target_faces or not candidate_faces:
    stop("No faces found!")


# ----------------------------------------------------------------------
# Find similar faces
# ----------------------------------------------------------------------

matches = azface_similar(client, target_faces, candidate_faces)
print_similar_results(target_faces, candidate_faces, matches)
