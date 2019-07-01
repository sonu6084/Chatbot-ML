from tkinter import *
from new_file import answer
from bot import wishMe
from female_voice import speak
from restaurant_info import get_rest_info
 
from tkinter import scrolledtext
 
b = Tk()
 
b.title("Chatbot")
 
b.geometry('1000x550')
topframe=Frame(b)
topframe.pack()
bottomframe=Frame(b)
bottomframe.pack(side=BOTTOM)
lbl=Label(topframe,text='FRIDAY',font=("Arial Bold", 20))
lbl.pack()
 
txt = scrolledtext.ScrolledText(topframe,width=100,height=28)
 
txt.pack()

lbl2=Label(bottomframe,text='USER > ',font=('Arial Bold',10))
lbl2.pack(side=LEFT)

txt1 = Entry(bottomframe,width=100)
txt1.pack(side=LEFT)

initial_query='LOC'

def clicked():
    txt.insert(INSERT,'\n')
    txt.insert(INSERT,'*'*80)
    ques='\nUSER > '+txt1.get()
    txt.insert(INSERT,ques)
    res=answer(txt1.get())
    if res!='restaurant':
        ans='\nBOT > '+res
        txt.insert(INSERT,ans)
    else:
        txt.insert(INSERT,'\nBOT >Enter your nearby location')
        initial_query='REST_NAME'

info_list=[]
def list_len(word):
    if len(info_list)<3:
        info_list.append(word)
    else :
        print(info_list)
def location():
    ques=txt1.get()
    txt.insert(INSERT,'\nUSER > '+ques)
    list_len(ques)
    txt.insert(INSERT,'\nBOT > Want to know info about specific restaurat or all restaurant in '+ques)


def restaurant_name():
    ques=txt1.get()
    txt.insert(INSERT,'\nUSER > '+ques)
    list_len(ques)
    txt.insert(INSERT,'\nBOT > What info you want to know about restaurants ')
    #ans2=get_rest_name(ques,ans1)

    
    
def final():
    ques=txt1.get()
    txt.insert(INSERT,'\nUSER > '+ques)
    list_len(ques)
    ans=get_rest_info(info_list)
    txt.insert(INSERT,'\nBOT > \n')
    txt.insert(INSERT ,ans)

    
wish,greet1=wishMe()

b4=Button(bottomframe,text='Enter Query',command=final)
b4.pack(side=RIGHT)

b3=Button(bottomframe,text='Enter Rest',command=restaurant_name)
b3.pack(side=RIGHT)

b2=Button(bottomframe,text='Enter Location',command=location)
b2.pack(side=RIGHT)

b1=Button(bottomframe, text="Enter",command =clicked)
b1.pack(side=RIGHT)

txt.insert(INSERT,'Your text goes here\n')
txt.insert(INSERT,'\nBOT > '+wish)

txt.insert(INSERT,"\nBOT > "+greet1)
 
b.mainloop()
