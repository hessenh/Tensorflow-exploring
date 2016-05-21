import httplib
import json

connection =httplib.HTTPConnection('api.football-data.org')
headers = { 'X-Auth-Token': '5cbb23f2d15746e8ad2b710363a9a997', 'X-Response-Control': 'minified' }
connection.request('GET', '/v1/soccerseasons/?season=2013', None, headers )
response = json.loads(connection.getresponse().read().decode())

import json
with open('data.json', 'w') as outfile:
    json.dump(response, outfile)

#for i in range(0,len(response)):
#	print response[i]