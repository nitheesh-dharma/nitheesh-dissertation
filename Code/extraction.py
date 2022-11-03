import sys,re
from urllib import response 
import xml.etree.ElementTree as ET
from unpywall.utils import UnpywallCredentials
from unpywall import Unpywall
import pandas as pd 
import json
import os
from scidownl import scihub_download
import requests
REVIEW_ID = "STD"
UnpywallCredentials('nitheeshdharmapalan@gmail.com')

study_info = dict()
with open("train_data.txt",'r') as f:
    review_file = f.read()

    with open("Reviews/Reviews/" + review_file
    ,'r',encoding='ISO-8859-1') as f1:
        review = f1.read()

    review_tree = ET.ElementTree(ET.fromstring(review))
    root = review_tree.getroot()
    #risk_of_bias = {}
    studies = set()
    
    for incl_study in root.iter('INCLUDED_STUDIES'):
        for study_data in incl_study: 
            #study_search_query = ""
            study_name = study_data.get("NAME")
            for study_ref in study_data.iter('REFERENCE'):
                title = study_ref.find('TI').text
                df = Unpywall.query(query = title,is_oa=True)
                if df is not None: 
                    doi_num = df['doi'].to_string(index=False)
                    try:
                        pdf_link = Unpywall.get_pdf_link(doi = doi_num)
                        response = requests.get(pdf_link)
                        file_path = os.path.join('Papers/Train',title + ".pdf")
                        with open(file_path,'wb') as f: 
                            f.write(response.content)
                        studies.add(study_name)
                        study_info[study_name] = [title,"Papers/Train/" + title + ".pdf"]
                    except:
                        doi_link = df['doi_url'].iloc[0]
                        author_name_retrieved = Unpywall.get_json(doi=doi_link)['z_authors'][0]['family']
                        author_name_actual = study_name.split()[0]
                        if author_name_actual == author_name_retrieved:
                            scihub_download(doi_link,paper_type = "doi",out = "Papers/Train" + title + ".pdf")
                            if os.path.isfile("Papers/" + title + ".pdf"):
                                studies.add(study_name)
                                study_info[study_name] = [title,"Papers/Train" + title + ".pdf"]

    for item in root.iter("QUALITY_ITEM"):
        pot_criteria = item.find('NAME').text.lower()
        data_item = item.find("QUALITY_ITEM_DATA")
        criteria = ""

        if "sequence" in pot_criteria: 
            criteria = "Random sequence generation"
        elif "concealment" in pot_criteria: 
            criteria = "Allocation concealment"
        elif "blinding" in pot_criteria and "outcome" in pot_criteria: 
            criteria = "Blinding of outcome assessment"
        elif "blinding" in pot_criteria: 
            criteria = "Blinding of participants and personnel"
        elif "outcome" in pot_criteria: 
            criteria = "Incomplete outcome data" 
        elif "selective" in pot_criteria: 
            criteria = "Selective reporting" 
        else: 
            criteria = "None"

        if criteria != "None":
            for data_item_entry in data_item.iter("QUALITY_ITEM_DATA_ENTRY"):
                study_id = data_item_entry.get("STUDY_ID") 
                study_name = study_id[4:].replace("-"," ")
                if study_name in studies:
                    result = data_item_entry.get("RESULT")
                    if result == "YES": 
                        risk = "Low risk"
                    elif result == "NO": 
                        risk = "High risk"
                    else: 
                        risk = "Unclear risk"

                    if len(study_info[study_name]) == 2: 
                        study_info[study_name].append({})
                    
                    study_info[study_name][2][criteria] = risk

    json_object = json.dumps(study_info)
    with open("train_data.json", "w") as outfile:
        outfile.write(json_object)
    #print(risk_of_bias)

    print(study_info)
    #print(study_link)