from female_voice import speak
import datetime
import wikipedia 
import webbrowser
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=5 and hour<12:
        wish="Good Morning!"
        #speak("Good Morning!")

    elif hour>=12 and hour<18:
        wish="Good Afternoon!"
        #speak("Good Afternoon!")

    elif hour>=18 and hour<21:
        wish="Good Evening!"
        #speak("Good Evening!")
    else:
        wish="Good Night!"
        #speak("Good Night!")

    greet1="I am FRIDAY .How may I help you ?"
    #speak("I am FRIDAY Sir. Please tell me how may I help you")

    return wish,greet1




def web_wiki(command):
    query = command.lower()
    if 'wikipedia' in query:
        #print('Searching Wikipedia...')
        #speak('Searching Wikipedia...')
        query = query.replace("wikipedia", "")
        results = wikipedia.summary(query, sentences=2)
        #print("According to Wikipedia")
        #speak("According to Wikipedia")
        #print(results)
        #speak(results)
        ans='According to Wikipedia\n'+results
        return ans

    elif 'open youtube' in query:
        webbrowser.open("youtube.com")

    elif 'mail' in query:
        webbrowser.open("gmail.com")
            
    elif 'open google' in query:
        webbrowser.open("google.com")

    elif 'open stackoverflow' in query:
        webbrowser.open("stackoverflow.com")   


    elif 'the time' in query:
        strTime = datetime.datetime.now().strftime("%H:%M:%S")    
        print(f"Sir, the time is {strTime}")
        speak('Sir, the time is ')

    else:
        pass
    
