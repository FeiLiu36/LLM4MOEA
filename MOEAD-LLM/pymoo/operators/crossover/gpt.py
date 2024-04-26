import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask
import json
import http.client
import re
import time


class GPT(Crossover):

    def __init__(self, n_new, **kwargs):
        super().__init__(10, 2, **kwargs)
        self.n_new = n_new


    def get_prompt(self,x,y,obj_p):
        # Convert given individuals to the desired string format
        pop_content = " "
        for i in range(len(x)):
            #pop_content+="point: <start>"+",".join(str(idx) for idx in x[i].tolist())+"<end> \n value: "+str(y[i])+" objective 1: "+str(obj_p[i][0])+" objective 2: "+str(obj_p[i][1])+"\n\n"
            pop_content+="point: <start>"+",".join(str(idx) for idx in x[i].tolist())+"<end> \n objective 1: "+str(round(obj_p[i][0],4))+" objective 2: "+str(round(obj_p[i][1],4))+"\n\n"
        
        prompt_content = "Now you will help me minimize "+str(len(obj_p[0]))+" objectives with "+str(len(x[0]))+" variables. I have some points with their objective values. The points start with <start> and end with <end>.\n\n" \
                        + pop_content \
                        +"Give me two new points that are different from all points above, and not dominated by any of the above. Do not write code. Do not give any explanation. Each output new point must start with <start> and end with <end>"
        return prompt_content

    def _do(self, _, X, Y,  debug_mode,model_LLM,key,out_filename,parents_obj, **kwargs):

        #x_scale = 1000.0
        y_p = np.zeros(len(Y))
        x_p = np.zeros((len(X),len(X[0][0])))
        for i in range(len(Y)):
            y_p[i]= round(Y[i][0][0],4)
            x_p[i] = X[i][0]

            x_p[i] = np.round((x_p[i] - _.xl)/(_.xu - _.xl),4)
        
        #x_p = x_scale*x_p

        sort_idx = sorted(range(len(Y)), key=lambda k: Y[k], reverse=True)
        x_p = [x_p[idx] for idx in sort_idx]
        y_p = [y_p[idx] for idx in sort_idx]
        obj_p = parents_obj[0][:10].get("F")
        obj_p = [obj_p[idx] for idx in sort_idx]

        
        prompt_content = self.get_prompt(x_p,y_p,obj_p)

        if debug_mode:
            print(prompt_content)
            print("> enter to continue")
            input()

        payload = json.dumps({
        #"model": "gpt-3.5-turbo",
        #"model": "gpt-4-0613",
        "model": model_LLM,
        "messages": [
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        "safe_mode": False
        })
        headers = {
        'Authorization': 'Bearer '+key,
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        'x-api2d-no-cache': 1
        }

        conn = http.client.HTTPSConnection("oa.api2d.site")
        #conn.request("POST", "/v1/chat/completions", payload, headers)

        retries = 5  # Number of retries
        retry_delay = 2  # Delay between retries (in seconds)
        while retries > 0:
            try:

                conn.request("POST", "/v1/chat/completions", payload, headers)

                res = conn.getresponse()
                data = res.read()

                # response_data = data.decode('utf-8')
                json_data = json.loads(data)
                #pprint.pprint(json_data)
                response = json_data['choices'][0]['message']['content']

                while(len(re.findall(r"<start>(.*?)<end>", response))<2):
                    conn = http.client.HTTPSConnection("oa.api2d.site")
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res = conn.getresponse()
                    data = res.read()

                    # response_data = data.decode('utf-8')
                    json_data = json.loads(data)

                    response = json_data['choices'][0]['message']['content']

                off_string1 = re.findall(r"<start>(.*?)<end>", response)[0]
                off1 = np.fromstring(off_string1, sep=",", dtype=float)

                off_string2 = re.findall(r"<start>(.*?)<end>", response)[1]
                off2 = np.fromstring(off_string2, sep=",", dtype=float)

                if out_filename != None:
                    filename=out_filename
                    file = open (filename,"a")
                    for i in range(len(x_p)):
                        for j in range(len(x_p[i])):
                            file.write("{:.4f} ".format(x_p[i][j]))
                        file.write("{:.4f} ".format(y_p[i]))
                    for i in range(len(off1)):
                        file.write("{:.4f} ".format(off1[i]))
                    for i in range(len(off1)):
                        file.write("{:.4f} ".format(off2[i]))
                    #file.write("{:.4f} {:.4f} {:.4f} {:.4f} \n".format(off1[0],off1[1],off[1][0][0],off[1][0][1]))
                    file.write("\n")
                    file.close

                off1[np.where(off1<0)] = 0.0
                off1[np.where(off1>1)] = 1.0
                off2[np.where(off2<0)] = 0.0
                off2[np.where(off2>1)] = 1.0
                off1 = np.array([[(off1*(_.xu - _.xl)+_.xl)]])
                off2 = np.array([[(off2*(_.xu - _.xl)+_.xl)]])
                off = np.append(off1,off2, axis=0)

                break

            except:
                print("Request failed !  ")
                retries -= 1
                if retries > 0:
                    print("Retrying in", retry_delay, "seconds...")
                    time.sleep(retry_delay)


        if debug_mode:
            print(response)
            print(off_string1)
            print(off_string2)
            print(off)
            print("> enter to continue")
            input()
        # print(off.shape)


        return off


class GPT_interface(GPT):

    def __init__(self, **kwargs):
        super().__init__(n_new=1, **kwargs)
