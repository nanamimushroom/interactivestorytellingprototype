from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
app = Flask(__name__)

page_index = 0
gm1 = pd.read_excel('./Yuan_DataStructure.xlsx', index_col='Index')
click_record = pd.read_csv('./click_record.csv')
gm_status = {}

gm_status['char_name'] = 'Unnamed Player'
gm_status['char_index'] = 0;
gm_status['self-confidence'] = 100
gm_status['happy'] = 100
gm_status['poll_1_index'] = 1;
gm_status['poll_2_index'] = 1;
#something to store the clicks
#something to define status

@app.route('/')
def show_startpage():

    return render_template('entername.html')

@app.route('/<input_int>', methods=['POST', 'GET'])
def render_gamepage(input_int):
    
    global click_record

    #firsr check whether the health bar is empty

    page_nr = int(input_int)
    template_type = gm1['template'][page_nr]
    html_name = gm1['template'][page_nr] + '.html'   #the int() conversion is important!

    if request.method == 'POST':
        #divide click event into multiple cases first

        result = request.form

        if 'CharNameInput' in result.keys():
            gm_status['char_name'] = request.form['CharNameInput']

        if 'ButtonNr' in result.keys(): #record the clicks
            click_record = click_record.append({'char_name': gm_status['char_name'], 'page_nr': result['PageNr'], 'button_nr': result['ButtonNr']}, ignore_index=True)

            click_record.to_csv('click_record.csv', index = False)

        #change the health bar

        if template_type == '3ball01':
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            return render_template(html_name, Button01Link = button_01_link, PageNr = page_nr)

        elif template_type == 'pollstart01':
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            button_02_link = '/' + str(int(gm1['button2link'][page_nr]))
            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            return render_template(html_name, text01 = text_01, Button01Link = button_01_link, Button02Link = button_02_link, PageNr = page_nr)

        elif template_type == 'pollstart02':
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            button_01_text = gm1['button1txt'][page_nr]
            return render_template(html_name, text01 = text_01, Button01Link = button_01_link, Button01Text = button_01_text, PageNr = page_nr)

        elif template_type == 'polltemplate01':
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            text_02 = str(int(gm_status['poll_1_index']))
            gm_status['poll_1_index'] += 1
            return render_template(html_name, text01 = text_01, text02 = text_02, Button01Link = button_01_link, PageNr = page_nr)

        elif template_type == 'polltemplate02':
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            text_02 = str(int(gm_status['poll_2_index']))
            gm_status['poll_2_index'] += 1
            return render_template(html_name, text01 = text_01, text02 = text_02, Button01Link = button_01_link, PageNr = page_nr)

        elif template_type == 'definition':
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            button_01_text = gm1['button1txt'][page_nr]
            return render_template(html_name, text01 = text_01, Button01Link = button_01_link, PageNr = page_nr, Button01Text = button_01_text)

        elif template_type == '3ball02' or template_type == '3ball03':
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            button_02_link = '/' + str(int(gm1['button2link'][page_nr]))
            button_03_link = '/' + str(int(gm1['button3link'][page_nr]))
            return render_template(html_name, Button01Link = button_01_link, Button02Link = button_02_link, Button03Link = button_03_link, PageNr = page_nr)

        elif template_type == 'startup01':
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            return render_template(html_name, text01 = text_01, Button01Link = button_01_link, PageNr = page_nr)

        #elif template_type == 'gameend':

        elif template_type == 'gameplot01':
            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            text_02 = gm1['text2'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            graph_01_link = gm1['graph1'][page_nr]
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            button_01_text = gm1['button1txt'][page_nr]
            return render_template(html_name, text01 = text_01, graph01 = graph_01_link, text02 = text_02, Button01Link = button_01_link, Button01Text = button_01_text, PageNr = page_nr)

        elif template_type == 'gameplot02':


            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            graph_01_link = gm1['graph1'][page_nr]
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            button_01_text = gm1['button1txt'][page_nr]
            button_02_link = '/' + str(int(gm1['button2link'][page_nr]))
            button_02_text = gm1['button2txt'][page_nr]

            return render_template(html_name, text01 = text_01, graph01 = graph_01_link, Button01Link = button_01_link, Button01Text = button_01_text, Button02Link = button_02_link, Button02Text = button_02_text, PageNr = page_nr)

        elif template_type == 'gameplot03':
            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            text_02 = gm1['text2'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            graph_01_link = gm1['graph1'][page_nr]
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))    #four links are the same
            button_01_text = gm1['button1txt'][page_nr]
            button_02_text = gm1['button2txt'][page_nr]
            button_03_text = gm1['button3txt'][page_nr]
            button_04_text = gm1['button4txt'][page_nr]

            return render_template(html_name, text01 = text_01, text02 = text_02, graph01 = graph_01_link, Button01Link = button_01_link, Button01Text = button_01_text, Button02Text = button_02_text, Button03Text = button_03_text, Button04Text = button_04_text,PageNr = page_nr)

        elif template_type == 'gameplot04':
            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            
            graph_01_link = gm1['graph1'][page_nr]
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            button_01_text = gm1['button1txt'][page_nr]
            button_02_link = '/' + str(int(gm1['button2link'][page_nr]))
            button_02_text = gm1['button2txt'][page_nr]
            button_03_link = '/' + str(int(gm1['button3link'][page_nr]))
            button_03_text = gm1['button3txt'][page_nr]

            return render_template(html_name, text01 = text_01, graph01 = graph_01_link, Button01Link = button_01_link, Button01Text = button_01_text, Button02Link = button_02_link, Button02Text = button_02_text, Button03Link = button_03_link, Button03Text = button_03_text,PageNr = page_nr)

        elif template_type == 'gameplot05':
            text_01 = gm1['text1'][page_nr].replace('$name', gm_status['char_name']).replace('\n', '<br/>')
            
            graph_01_link = gm1['graph1'][page_nr]
            button_01_link = '/' + str(int(gm1['button1link'][page_nr]))
            button_01_text = gm1['button1txt'][page_nr]
            return render_template(html_name, text01 = text_01, graph01 = graph_01_link, Button01Link = button_01_link, Button01Text = button_01_text, PageNr = page_nr)


if __name__ == '__main__':
   app.run()