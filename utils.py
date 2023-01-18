from resources import config, tokenizer, model
from pydantic import BaseModel

import os
import sys
import re
import json
import requests
import urllib.request
import urllib.parse
import urllib.error

class Description(BaseModel):
    description: str

def get_code(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    codes = [config.id2label[ids] for ids in output.logits.detach().cpu().numpy()[
        0].argsort()[::-1][:5]]
    return codes


def translate(to_translate, to_language="en", language="es"):
    # la base_url es la url de la api de google translate a la que se le va a enviar 
    # la palabra a traducir y el idioma de la palabra 
    base_url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl={}&tl={}&dt=t&q={}" 
    # to_translate es la palabra a traducir 
    to_translate = urllib.parse.quote(to_translate)
    # to_language es el idioma de la palabra a traducir
    url = base_url.format(language, to_language, to_translate)
    # se hace la peticion a la api de google translate
    response = urllib.request.urlopen(url)
    # se obtiene la respuesta en formato json
    result = json.loads(response.read().decode())
    # se obtiene la traduccion de la palabra
    return result[0][0][0]


def traducir(palabra):
    palabra = translate(palabra)
    return palabra