#!/usr/bin/env python3
"""
Simple HTTP server for Unsplash DB semantic search
Assumes moat.com vectors are in the directory moat-dataset/
urls:
    wget https://moat-dataset.s3-us-west-2.amazonaws.com/features.npy
    wget https://moat-dataset.s3-us-west-2.amazonaws.com/photo_ids.csv

Assumes unsplash.com vectors are in the directory moat-dataset/
urls:
    wget https://unsplash-dataset.s3-us-west-2.amazonaws.com/features.npy
    wget https://unsplash-dataset.s3-us-west-2.amazonaws.com/photo_ids.csv

Usage::
    python ./server.py [<port>]

Testing::
   curl -H "Content-Type: application/json" -X POST -d '{"prompt":["The quick brown fox"], "moat_selected": ["197_859", "23_345"], "num": 3}' http://127.0.0.1:8000/api
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from json import dumps
import logging
import json

import clip
import torch
import os
import pandas as pd
import numpy as np

print("Loading features...")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the photo IDs from moat
photo_ids = pd.read_csv("unsplash-dataset/photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])

# Load the features vectors
photo_features = np.load("unsplash-dataset/features.npy")

# Load the photo IDs from moat
moat_ids = pd.read_csv("moat-dataset/photo_ids.csv")
moat_ids = list(moat_ids['photo_id'])

# Load the features vectors from moat
moat_features = np.load("moat-dataset/features.npy")


# Convert features to Tensors: Float32 on CPU and Float16 on GPU
if device == "cpu":
  photo_features = torch.from_numpy(photo_features).float().to(device)
  moat_features = torch.from_numpy(moat_features).float().to(device)
else:
  photo_features = torch.from_numpy(photo_features).to(device)
  moat_features = torch.from_numpy(moat_features).to(device)

# Print some statistics
print(f"Unsplash photos loaded: {len(photo_ids)}")
print(f"Moat photos loaded: {len(moat_ids)}")


class S(BaseHTTPRequestHandler):
    timeout = 15
    def _send_cors_headers(self):
      """ Sets headers required for CORS """
      self.send_header("Access-Control-Allow-Origin", "*")
      self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
      self.send_header("Access-Control-Allow-Headers", "x-api-key,Content-Type")

    def do_OPTIONS(self):
       self.send_response(200)
       self._send_cors_headers()
       self.end_headers()

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

        help = "do POST request to route /api with JSON object {'prompt':[]} with semantic query"
        self._set_response()
        self.wfile.write(help.encode('utf-8'))


    def do_POST(self):

        error = 'JSON object misses "prompt" key'

        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        objectDict = json.loads(post_data.decode('utf-8'))
       
        message = {}
        print(f"received {objectDict}") 

        num = 3
        moat_selected = []
        if ("num" in objectDict) and isinstance(objectDict["num"], int):
           num = objectDict["num"]

        if ("moat_selected" in objectDict) and isinstance(objectDict["moat_selected"], list):
           moat_selected = objectDict["moat_selected"]

        if ("prompt" in objectDict) and isinstance(objectDict["prompt"], list):
           # ignoring null length lists etc
           best_images = search_unsplash(objectDict["prompt"][0], photo_features, photo_ids, num)   
           
           for ad in moat_selected:
              try:
              #search for the moat adm
                 ad_index = moat_ids.index(ad)
                 print('The index of', ad, 'in the list is:', ad_index)
                 best_images += search_unsplash_with_moat_index(ad_index, photo_features, photo_ids, moat_features, num)
              except ValueError:
                 print(f'ad {ad} requested but not found')
              
           message =  json.dumps(best_images)
        else: 
           message = error        
        
        self.wfile.write(message.encode('utf-8'))

def encode_search_query(search_query):
  with torch.no_grad():
    # Encode and normalize the search query using CLIP
    text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

  # Retrieve the feature vector
  return text_encoded

def find_best_matches(text_features, photo_features, photo_ids, results_count=3):
   # Compute the similarity between the search query and each photo using the Cosine similarity
   similarities = (photo_features @ text_features.T).squeeze(1)
   # Sort the photos by their similarity score
   best_photo_idx = (-similarities).argsort()
   # Return the photo IDs of the best matches
   return [photo_ids[i] for i in best_photo_idx[:results_count]]

def search_unsplash(search_query, photo_features, photo_ids, results_count=3):
   # Encode the search query
   text_features = encode_search_query(search_query)
   # Find the best matches
   best_photo_ids = find_best_matches(text_features, photo_features, photo_ids, results_count)
   # Display the best photos
   for photo_id in best_photo_ids:
      print(photo_id)

   return best_photo_ids

def search_unsplash_with_moat_index(ad_index, photo_features, photo_ids, moat_features, results_count=3):

  # Find the best matches
  best_photo_ids = find_best_matches(torch.reshape(moat_features[ad_index], (1,-1)), photo_features, photo_ids, results_count)

  # Display the best photos
  return best_photo_ids

def run(server_class=HTTPServer, handler_class=S, port=8000):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    httpd.timeout = 10
    logging.info('Starting httpd...\n')
    print(f"Ready to run")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    from sys import argv
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()

