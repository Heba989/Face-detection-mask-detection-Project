import requests

#requests to pmake prediction
url = 'http://127.0.0.1:5000/predict'
files = {'image': open(r'C:Users/user/Desktop/ICT/Python/11751.png','rb')}
x = requests.post(url, files=files)
print(x.text)

# >>> files = {'image': open(r'C:\Users\user\Desktop\ICT\Python\11751.png','rb')}
# >>> x = requests.post('http://127.0.0.1:5000/predict', files=files)
#request to retrieve data as csv file 
rep= requests.get('http://127.0.0.1:5000/retrieve',{'time':'flag="1"'})
print(rep.text)
