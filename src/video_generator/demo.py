import requests


data = {'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
        'response_type': 'cloud_iam',
        'apikey': "kKa3P-xl9gjjECYCAF0872FnvW03wIiH9fN7htuybP4o"}

headers = {'accept': "application/json",
                'authorization': "Basic Yng6Yng=",
                'cache-control': "no-cache",
                'Content-Type': "application/x-www-form-urlencoded"}


response = requests.post('https://iam.cloud.ibm.com/identity/token', data=data, headers=headers, verify=True)

# print(response.json())


url = "https://s3.us-south.cloud-object-storage.appdomain.cloud/shortsai/hello.txt"

response = requests.get(url=url, headers=headers)

print(response.request.path_url)