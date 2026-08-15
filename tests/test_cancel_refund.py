import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm.agent import LLMAgent
import json

def run():
    agent=LLMAgent()
    t='I cancelled my policy but haven\'t received money'
    res=agent.analyze_ticket(t, category='Policy Changes')
    print(json.dumps(res, indent=2))

if __name__=='__main__':
    run()
