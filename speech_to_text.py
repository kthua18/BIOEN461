import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source) #, timeout=1)

# recognize speech using Sphinx
try:
    print("Google thinks you said: " + r.recognize_google(audio))
    speech = r.recognize_google(audio)
except sr.UnknownValueError:
    print("Google could not understand audio")
except sr.RequestError as e:
    print("Google error; {0}".format(e))


list_of_phrases = ["What's in my view", "What can I see", "What's in my field of vision"]

#here should be a map with key to values, where values are tied to actions

#now I need to map fucntions that tell us what each thing does
