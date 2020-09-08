## Create Authentication textfile ##
from pydrive.auth import GoogleAuth
gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
gauth.SaveCredentialsFile("mycreds.txt")
