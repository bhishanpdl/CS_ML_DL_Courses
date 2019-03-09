import pandas as pd
import pandasql as pdsql


def max_temp_aggregate_by_fog(filename):
    '''
    This function should run a SQL query on a dataframe of
    weather data.  The SQL query should return two columns and
    two rows - whether it was foggy or not (0 or 1) and the max
    maxtempi for that fog value (i.e., the maximum max temperature
    for both foggy and non-foggy days).  The dataframe will be 
    titled 'weather_data'. You'll need to provide the SQL query.
    
    You might also find that interpreting numbers as integers or floats may not
    work initially.  In order to get around this issue, it may be useful to cast
    these numbers as integers.  This can be done by writing cast(column as integer).
    So for example, if we wanted to cast the maxtempi column as an integer, we would actually
    write something like where cast(maxtempi as integer) = 76, as opposed to simply 
    where maxtempi = 76.
    
    You can see the weather data that we are passing in below:
    https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/weather_underground.csv
    '''
    weather_data = pd.read_csv(filename)

    q = """
    SELECT fog, MAX(cast(maxtempi as integer)) 
    FROM weather_data 
    GROUP BY fog;
    """
    
    #Execute your SQL command against the pandas frame
    foggy_days = pdsql.sqldf(q.lower(), locals())
    return foggy_days


filename = '../data/weather_underground.csv'
n = max_temp_aggregate_by_fog(filename)
print(n)