from datetime import datetime
import copy
import sys
from util import sentinel_util

def filter_products(product_list, tileId=None, cloudCover_min=None, cloudCover_max=None, productType=None):
    #TODO parameter list could actually be an arbitrary amount with keys being the same as in the documentation...
    product_list = product_list["value"]
    result = []
    #iterates through a list of dictionaries
    for elem in product_list:
        #this gives a list of dictionaries where each dictionary corresponds to a specific attribute
        attributes = elem["Attributes"]
        if tileId is not None:
            tileId_idx = sentinel_util.get_attribute_index(attributes, "tileId")
            if attributes[tileId_idx]["Value"] == tileId:
                pass
            else:
                continue
        if productType is not None:
            productType_idx = sentinel_util.get_attribute_index(attributes, "productType")
            if attributes[productType_idx]["Value"] == productType:
                pass
            else:
                continue
        if cloudCover_min is not None:
            cloudCover_idx = sentinel_util.get_attribute_index(attributes, "cloudCover")
            if attributes[cloudCover_idx]["Value"] >= cloudCover_min:
                pass
            else:
                continue
        if cloudCover_max is not None:
            cloudCover_idx = sentinel_util.get_attribute_index(attributes, "cloudCover")
            if attributes[cloudCover_idx]["Value"] <= cloudCover_max:
                pass
            else:
                continue
        result.append(elem)
    return result

def get_date_from_name(id_name: str) -> datetime:
    str_datetime = id_name.split("_")[2]
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

def keep_highest_N(ids):
    #pair same products and only keep the one with the highest postprocessor version
    result = []
    skipper = []
    #list of dictionaries
    for i in range(len(ids)):
        if i not in skipper:
            duplicates, duplicate_idx  = find_duplicate(ids[i], ids[i+1:])
            duplicate_idx = [idx+i+1 for idx in duplicate_idx]  # as the given idx list is based on the list without the first i elements we have to readd i. And +1 because indices are always one element less than the length
            skipper += duplicate_idx
            result.append(find_highest_processor(duplicates))
    return result


def find_highest_processor(duplicates):
    highest_idx = 0
    highest_ver = -1
    for idx, elem in enumerate(duplicates):
        atts = elem["Attributes"]
        processor_idx = sentinel_util.get_attribute_index(atts, "processorVersion")
        if float(atts[processor_idx]["Value"]) >= float(highest_ver):
            highest_ver = atts[processor_idx]["Value"]
            highest_idx = idx
    return duplicates[highest_idx]



def find_duplicate(id, id_list):
    #find duplicates which only differ in processor version
    # removes the NXXX in the middle
    compare = "_".join(id["Name"].split("_")[:3]) + "_".join(id["Name"].split("_")[4:-1])
    identicals = [id]
    identical_idx = []
    for idx, elem in enumerate(id_list):
        list_compare = "_".join(elem["Name"].split("_")[:3]) + "_".join(elem["Name"].split("_")[4:-1])
        if compare == list_compare:
            identicals.append(elem)
            identical_idx.append(idx)
    return identicals, identical_idx