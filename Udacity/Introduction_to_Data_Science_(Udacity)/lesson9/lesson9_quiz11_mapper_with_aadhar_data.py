import sys
import string
import logging


# from util import mapper_logfile
# logging.basicConfig(filename=mapper_logfile, format='%(message)s',
#                     level=logging.INFO, filemode='w')

HEADER_LINE ="Registrar,Enrolment Agency,State,District,Sub District,Pin Code,Gender,Age,Aadhaar generated,Enrolment Rejected,Residents providing email,Residents providing mobile number"
HEADER_LIST = HEADER_LINE.split(',')
DISTRICT_INDEX = HEADER_LIST.index("District")
AADHAAR_GENERATED_INDEX = HEADER_LIST.index("Aadhaar generated")

def mapper():
    for line in sys.stdin:
        # Dirty check so we dont evaluate a header line
        if "Registrar,Enrolment Agency,State," in line:
            continue

        data = line.split(",")
        district = data[DISTRICT_INDEX]
        aadhaar_generated = data[AADHAAR_GENERATED_INDEX]

        if district and aadhaar_generated:
            print ("{0}\t{1}".format(district,aadhaar_generated))

# Run the code
mapper()