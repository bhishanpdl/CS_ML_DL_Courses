import sys
import logging

# from util import reducer_logfile
# logging.basicConfig(filename=reducer_logfile, format='%(message)s',
#                     level=logging.INFO, filemode='w')

def reducer():

    aadhaar_generated_counts = {}

    for line in sys.stdin:
        district, number_generated = line.split("\t")

        if not district and number_generated:
            continue

        number_generated = int(number_generated)

        if district in aadhaar_generated_counts.keys():
            aadhaar_generated_counts[district] += number_generated
        else:
            aadhaar_generated_counts[district] = number_generated

    for key in aadhaar_generated_counts:
        print ("{0}\t{1}".format(key, aadhaar_generated_counts[key]))



reducer()