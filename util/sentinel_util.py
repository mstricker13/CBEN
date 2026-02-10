from datetime import datetime

def get_attribute_index(attributes, att_name):
    #find the index of dictionary with name att_name in a list of attributes
    for idx, attr in enumerate(attributes):
        if attr["Name"] == att_name:
            return idx