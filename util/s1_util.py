from util import sentinel_util
import re
import sys
from datetime import datetime
import shapely

def filter_products(product_list, operationalMode=None, productClass=None, productType=None, polarisationChannels=None, blacklist=None):
    #TODO parameter list could actually be an arbitrary amount with keys being the same as in the documentation...
    product_list = product_list["value"]
    result = []
    #iterates through a list of dictionaries
    for elem in product_list:
        attributes = elem["Attributes"]
        if elem["Id"] in blacklist:
            continue
        if operationalMode is not None:
            operationalMode_idx = sentinel_util.get_attribute_index(attributes, "operationalMode")
            if attributes[operationalMode_idx]["Value"] == operationalMode:
                pass
            else:
                continue
        if productClass is not None:
            productClass_idx = sentinel_util.get_attribute_index(attributes, "productClass")
            if attributes[productClass_idx]["Value"] == productClass:
                pass
            else:
                continue
        if productType is not None:
            productType_idx = sentinel_util.get_attribute_index(attributes, "productType")
            if attributes[productType_idx]["Value"] == productType:
                pass
            else:
                continue
        if polarisationChannels is not None:
            polarisationChannels_idx = sentinel_util.get_attribute_index(attributes, "polarisationChannels")
            if attributes[polarisationChannels_idx]["Value"] == polarisationChannels:
                pass
            else:
                continue
        result.append(elem)
    return result

def get_date_from_name(id_name: str) -> datetime:
    str_datetime = id_name.split("_")
    for elem in str_datetime:
        date_re = re.compile(r'[0123456789]T[0123456789]')
        if date_re.search(elem):
            str_datetime = elem
            break
    dateformat = "%Y%m%dT%H%M%S"
    return datetime.strptime(str_datetime, dateformat).date()

def get_temporal_closest(date, products):
    closes_idx = 0
    closesdate = None
    for idx, elem in enumerate(products):
        elem_date = get_date_from_name(elem["Name"])
        if closesdate is None:
            closesdate = abs(date-elem_date)
            closes_idx = idx
        else:
            if abs(date-elem_date) < closesdate:
                closesdate = abs(date-elem_date)
                closes_idx = idx
    return products[closes_idx]