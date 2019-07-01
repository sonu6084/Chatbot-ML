from bot import web_wiki
from get_answer import get_user_input
def answer(a):
    if a=='quit':
        ans="Pleasure talking to you. See you again"
        
    else:
        if 'wikipedia' in a.lower():
            ans=web_wiki(a)
        else:
            web_wiki(a)
            ans=get_user_input(a)
    return ans


