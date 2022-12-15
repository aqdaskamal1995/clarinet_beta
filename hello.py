import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import webbrowser

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# embedding_data_path = '/Users/aqdaskamal/Desktop/IDEP Prototype/df_embeddings.csv'
# embedding_data = pd.read_csv(embedding_data_path)
obj = pd.read_pickle(r'course_tensor_dict.pickle')
undergrad_tensor = pd.read_pickle(r'undergrad_tensor_dict.pickle')

person_email_dict = {'Connie Wang': 'connie@mde.harvard.edu',
'Austin Ledzian':' austin@mde.harvard.edu',
'Dishi Gautam': 'dishi@mde.harvard.edu',
'Ana Merla':'ana@mde.harvard.edu',
'Cedric-Pascal Sommer':  'cedric@mde.harvard.edu',
'Hana Yamaki': 'hana@mde.harvard.edu',
'Riad El Soufi':' riad@mde.harvard.edu',
'Jae An': 'jae@mde.harvard.edu',
'Felly Liang': 'felly@mde.harvard.edu',
'Rebecca Brand': 'rebecca@mde.harvard.edu',
'Luke Reeve': 'luke@mde.harvard.edu',
'Daniel Feist': 'daniel@mde.harvard.edu',
'Arthur van Havre': 'arthur@mde.harvard.edu',
'Steven Morse': 'steven@mde.harvard.edu',
'Jenny Jiang': 'jenny@mde.harvard.edu',
'Jiwon Woo': 'jiwon@mde.harvard.edu',
'Jade Wu': 'jade@mde.harvard.edu',
'Gavin Jiao': 'gavin@mde.harvard.edu',
'Jon Chinen': 'jon@mde.harvard.edu',
'Mimi Kigawa': 'mimi@mde.harvard.edu',
'Aqdas Kamal': 'aqdas@mde.harvard.edu'}


person_emoji_dict = {'Connie Wang':'🙎🏼',
'Austin Ledzian': '🙎‍♂️',
'Dishi Gautam': '🙎🏼',
'Ana Merla': '🙎🏼',
'Cedric-Pascal Sommer':'🙎‍♂️',
'Hana Yamaki': '🙎🏼',
'Riad El Soufi':'🙎‍♂️',
'Jae An':'🙎‍♂️',
'Felly Liang': '🙎🏼',
'Rebecca Brand': '🙎🏼',
'Luke Reeve':'🙎‍♂️',
'Daniel Feist':'🙎‍♂️',
'Arthur van Havre':'🙎‍♂️',
'Steven Morse':'🙎‍♂️',
'Jenny Jiang': '🙎🏼',
'Jiwon Woo': '🙎🏼',
'Jade Wu': '🙎🏼',
'Gavin Jiao':'🙎‍♂️',
'Jon Chinen':'🙎‍♂️',
'Mimi Kigawa': '🙎🏼',
'Aqdas Kamal':'🙎‍♂️'}

st.title('Clarinet β')
st.subheader("I'm just in β right now so be nice to me!")

# course_param= st.slider('How much does what courses they')
# undergrad_param = st.slider('undergrad_param')

course_param = 1
undergrad_param = 2

# course_threshold = st.number_input('course_threshold')
# undergrad = st.number_input('undergrad_threshold')

query = st.text_input(label= 'Type your query', placeholder ='E.g I am looking for courses in Machine Learning and Data Science?')
query_embedding = embedder.encode(query, convert_to_tensor=False)

run_query = st.button('Ask!')

people_course_table = pd.read_csv('people_course_table.csv')
people_academic_table =pd.read_csv('person_academic.csv')

if run_query:
    df_embedding_score = pd.DataFrame()
    course_list = []
    score_list = []

    df_embedding_score_education = pd.DataFrame()
    person_list = []
    score_list_education = []

    for course in obj.keys():
        course_list.append(course)
        score_list.append(util.cos_sim(obj[course],query_embedding)[0][0].item())
    
    for person in undergrad_tensor.keys():
        person_list.append(person)
        score_list_education.append(util.cos_sim(undergrad_tensor[person],query_embedding)[0][0].item())

   
    df_embedding_score_education['Person'] = person_list
    df_embedding_score_education['Score_Education'] = score_list_education
    df_embedding_score['Course'] = course_list
    df_embedding_score['Score'] = score_list

    df_score = people_course_table.merge(df_embedding_score, left_on='Course', right_on='Course', how='left')
    df_score = df_score.merge(df_embedding_score_education, left_on='Name', right_on='Person', how='left')
    # st.write(df_score)
    df_score['Mean Score'] =  course_param*df_score['Score']+undergrad_param*df_score['Score_Education']
    score_course = df_score.sort_values("Mean Score", ascending=False).iloc[1]['Score']
    score_course_education= df_score.sort_values("Mean Score", ascending=False).iloc[1]['Score_Education']

    df_score = df_score.merge(people_academic_table, left_on='Name', right_on='Name', how='left')

    top_course = df_score.sort_values("Mean Score", ascending=False).iloc[0]['Course']
    top_course_desc = df_score.sort_values("Mean Score", ascending=False).iloc[0]['Description']
    relevant_people = df_score.sort_values("Mean Score", ascending=False).iloc[0]['Name']
    undergrad = df_score.sort_values("Mean Score", ascending=False).iloc[0]['Academic Background']
    # undergrad = df_score.sort_values("Mean Score", ascending=False).iloc[0]['Name']
   
   


    # top_course = df_embedding_score.sort_values("Score", ascending=False).iloc[0]['Course']
    # top_course_desc = people_course_table[people_course_table['Course']==top_course]['Description'].iloc[0]
    # relevant_people = people_course_table[people_course_table['Course']==top_course]['Name'].tolist()[0]



    st.write("Here's who you should speak to")
    col1, col2  = st.columns([1.5, 1])
    with col1:
        st.success(relevant_people, icon=person_emoji_dict[relevant_people])
    with col2:
        with st.expander("Contact   📧"):
            st.write(person_email_dict[relevant_people])

    # with col3:
    #     button_1_email = st.button(label='https://icons8.com/icon/85088/whatsapp',  key = 'Button2_Email')
    # with col4:
    #     button_1_email = st.button(label='📧',  key = 'Button3_Email')
    #     # button_1 = st.button(label='Connect',  key = 'Button1')
    #     # if button_1:
    #     #     # webbrowser.open_new_tab('cedricsommer@mde.harvard.edu')s
    #     #     st.write('cedricsommer@mde.harvard.edu')
    #     st.markdown(f'''<a href={url}><button style="background-color:White;">Connect</button></a>''',unsafe_allow_html=True)



    # for person in relevant_people:
    #     st.success(person, icon='👨🏽‍🎓')
    # st.title('Why?')
    with st.expander("See explanation"):
        if score_course > score_course_education:
            st.write('Because they took ' + top_course)
            st.write('What is ' +top_course +'?')
            st.write(top_course_desc)
        else:
            st.write('Because they majored in ' + undergrad+ '  in undergrad')
        
        
        

    

    df_score_filt = df_score[df_score['Name']!=relevant_people]
    score_course_2= df_score_filt.sort_values("Mean Score", ascending=False).iloc[1]['Score']
    score_course_education_2= df_score_filt.sort_values("Mean Score", ascending=False).iloc[1]['Score_Education']
    score_course_2= df_score_filt.sort_values("Mean Score", ascending=False).iloc[1]['Score']
    top_course_2 = df_score_filt.sort_values("Mean Score", ascending=False).iloc[1]['Course']
    top_course_desc_2 = df_score_filt.sort_values("Mean Score", ascending=False).iloc[1]['Description']
    relevant_people_2 = df_score_filt.sort_values("Mean Score", ascending=False).iloc[1]['Name']
    undergrad_2 = df_score_filt.sort_values("Mean Score", ascending=False).iloc[1]['Academic Background']

    # top_course_2 = df_embedding_score.sort_values("Score", ascending=False).iloc[1]['Course']
    # top_course_desc_2 = people_course_table[people_course_table['Course']==top_course_2]['Description'].iloc[0]
    # relevant_people_2 = people_course_table[people_course_table['Course']==top_course_2]['Name'].tolist()[0]
    st.write("You could also connect with...")
    col1, col2  = st.columns([1.5, 1])
    with col1:
        st.success(relevant_people_2, icon=person_emoji_dict[relevant_people_2])
    with col2:
        with st.expander("Contact   📧"):
            st.write(person_email_dict[relevant_people_2])
    # for person in relevant_people_2:
    #     st.success(person, icon='👨🏽‍🎓')
    with st.expander("See explanation"):
        if score_course_2 > score_course_education_2:
            st.write('Because they took ' + top_course_2)
            st.write('What is ' +top_course_2 +'?')
            st.write(top_course_desc_2)
        
        else:
            st.write('Because they majored in ' + undergrad_2+ '  in undergrad')
        
