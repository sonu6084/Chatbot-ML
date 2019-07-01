import pyttsx3
engine=pyttsx3.init()
sound=engine.getProperty('voices')
engine.setProperty('voice',sound[1].id)
def speak(words):
    engine.say(words)
    engine.runAndWait()


