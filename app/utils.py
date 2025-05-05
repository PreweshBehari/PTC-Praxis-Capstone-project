##########################################################################################
# Imports
##########################################################################################
import hashlib

from datetime import datetime


##########################################################################################
# Utils
##########################################################################################
def create_unique_value(combined_keys):
    unique_hash = hashlib.md5(combined_keys.encode()).hexdigest()
    return unique_hash

def human_readable_date(input_date, input_date_format="%Y-%m-%d", output_date_format="%d %B %Y"):
    try:
        if isinstance(input_date, str):
            date_object = datetime.strptime(input_date, input_date_format)
        else:
            date_object = input_date

        human_readable = date_object.strftime(output_date_format)
    except Exception as e:
        print(f"\nHuman Readable Date exception: {e}")
        return input_date

    return human_readable