import sys
from abc import ABC
from pathlib import Path
import zipfile

import geopandas as gpd
import util.meta_json_util
import util.project_util

import requests

import numpy as np

#TODO might add decorators to enforce inclusion of certain methods/attributes
class Odata(ABC):
    """
    Custom interface to Odata to work within my project architecture.
    """
    web_page = "https://documentation.dataspace.copernicus.eu/APIs/OData.html"

    def __init__(self, user: str = None, password: str = None, meta_cfg: Path = None) -> str:
        """
        Login into copernicus OData framework.

        Implementation is based on documentation: https://documentation.dataspace.copernicus.eu/APIs/OData.html

        Returns
        --------------------------------
        The access token
        """
        # Try to login Login to the odata api.
        meta_cfg = meta_cfg if meta_cfg is not None else util.project_util.get_meta_info_loc()
        user = util.meta_json_util.extract_odata_user(meta_cfg) if user is None else user
        password = util.meta_json_util.extract_odata_password(meta_cfg) if password is None else password
        return Odata.__login(user, password)

    @staticmethod
    def __login(user: str, password: str) -> str:
        data = {
            "client_id": "cdse-public",
            "username": user,
            "password": password,
            "grant_type": "password",
        }
        try:
            r = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data=data,
            )
            r.raise_for_status()
        except Exception as e:
            raise Exception(
                f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
            )
        return (r.json()["access_token"], r.json()["refresh_token"])
    
    @staticmethod
    def refresh(refresh_token):
        data = {
            "grant_type": "refresh_token",
            "client_id": "cdse-public",
            "refresh_token": refresh_token,
        }
        try:
            r = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data=data,
            )
            r.raise_for_status()
        except Exception as e:
            raise Exception(
                f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
            )
        return (r.json()["access_token"], r.json()["refresh_token"])

    @staticmethod
    def get_products(constellation, footprint, begin, end):
        #Note according to documentation https://documentation.dataspace.copernicus.eu/APIs/OData.html
        #Coordinates must be given in EPSG 4326

        #if Footprint is a path
        if isinstance(footprint, Path):
            area = gpd.read_file(footprint)
            xx, yy = area["geometry"][0].exterior.coords.xy
        else:
            xx = footprint[0]
            yy = footprint[1]
        query = (f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
                 f"?$filter=Collection/Name eq '{constellation}' and "
                 f"ContentDate/Start gt {begin}T00:00:00.000Z and "
                 f"ContentDate/Start lt {end}T23:59:59.000Z and "
                 f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({xx[0]} {yy[0]},{xx[1]} {yy[1]},{xx[2]} {yy[2]},{xx[3]} {yy[3]},{xx[4]} {yy[4]}))')&$expand=Attributes&$top=1000"
            )
        return requests.get(query).json()
    
    @staticmethod
    def download(id, token, out):
        
        print(f"Downloading {out} with ID {id}")

        out.parent.mkdir(parents=True, exist_ok=True)

        url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"

        headers = {"Authorization": f"Bearer {token}"}

        # Create a session and update headers
        session = requests.Session()
        session.headers.update(headers)

        # Perform the GET request
        response = session.get(url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            with open(out, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
            print(response.text)

        with zipfile.ZipFile(out, 'r') as zip_ref:
            extract_dir = out.parent / "product"
            zip_ref.extractall(extract_dir)
