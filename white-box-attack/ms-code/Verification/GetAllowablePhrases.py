########### Python 3.2 #############
import http.client, urllib.request, urllib.parse, urllib.error, base64

headers = {
    # Request headers
    'Ocp-Apim-Subscription-Key': '3c2c75effb044a1297d6a9d27345e608',
}

params = urllib.parse.urlencode({
    "locale" : "zh-CN"
})

try:
    conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
    #conn.request("GET", "/spid/v1.0/verificationPhrases?locale={locale}&%s" % params, "{body}", headers)
    conn.request("GET", "/spid/v1.0/verificationPhrases?locale=en-us&%s", "{body}", headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

####################################

