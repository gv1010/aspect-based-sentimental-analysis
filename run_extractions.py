from simpletransformers.ner import NERModel, NERArgs
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import os
import xml.etree.ElementTree as ET
import pandas as pd
import json

tree = ET.parse('./Restaurants_Train_v2.xml')
root = tree.getroot()
# Parse the XML data
# root = ET.fromstring(xml_data)
# Function to extract positions

def extract_positions(root):
    data = []
    aspect_term_cat_list = []
    sentences = root.findall('sentence')
    for sentence in sentences:
        new_dict = {}
        sentence_id = sentence.get('id')
        text = sentence.find('text').text
        new_dict["sentence_id"] = sentence_id
        new_dict["text"] = text
        new_dict["aspectTerms"] = []
        new_dict["aspectCategory"] = []
        d = [sentence_id, text]

        aspect_terms = sentence.find('aspectTerms')
        aspect_terms_list_sent = []
        if aspect_terms is not None:

            for aspect_term in aspect_terms.findall('aspectTerm'):
                aspect_term_dict = {}
                term = aspect_term.get('term')
                term_from = int(aspect_term.get('from', 0))
                term_to = int(aspect_term.get('to', 0))
                polarity = aspect_term.get('polarity')

                aspect_term_dict["term"] = term
                aspect_term_dict["term_from"] = term_from
                aspect_term_dict["term_to"] = term_to
                aspect_term_dict["polarity"] = polarity
                data.append( d + [term, term_from, term_to, polarity, "", ""])
                new_dict["aspectTerms"].append(aspect_term_dict)
                aspect_terms_list_sent.append(term)
                # print(f"Sentence ID: {sentence_id}, Term: '{term}', Start: {term_from}, End: {term_to}, Polarity: {polarity}")

        aspect_categories = sentence.find('aspectCategories')
        aspect_categ_list_sent = []
        if aspect_categories is not None:
            for aspect_category in aspect_categories.findall('aspectCategory'):
                aspect_cat_dict = {}
                category = aspect_category.get('category')
                polarity = aspect_category.get('polarity')

                aspect_cat_dict["category"] = category
                aspect_cat_dict["polarity"] = polarity

                data.append( d + ["", 0, 0, "", category, polarity])
                new_dict["aspectCategory"].append(aspect_cat_dict)
                aspect_categ_list_sent.append(category)

                # For categories, we're not given start and end positions, so we consider the entire sentence.
                # print(f"Sentence ID: {sentence_id}, Category: '{category}', Polarity: {polarity}, Text: '{text}'")
        aspect_term_cat_list.append((int(sentence_id), text, aspect_terms_list_sent, aspect_categ_list_sent))
        # data.append(new_dict)
    columns = ["sent_id", "text", "term", "term_from", "term_to", "term_polarity", "category", "category_polarity"]
    df =  pd.DataFrame(data, columns=columns)
    return df, aspect_term_cat_list

def get_polarity(polarity_model, sentence, aspect_list):
    polarity_dict = []
    for aspect in aspect_list:
        polarity_dict.append(polarity_model.predict([sentence +" [SEP] "+ aspect])[0][0])
    return polarity_dict


def get_category(category_model, sentence, aspect_list):
    category_dict = []
    for aspect in aspect_list:
        category_dict.append(category_model.predict([sentence +" [SEP] "+ aspect])[0][0])
    return category_dict


def gets_output_on_sentence(sentence):
    aspect_predictions, _ = aspect_model.predict([sentence])
    tokens_list = aspect_predictions[0]
    aspect_list_sent = combine_aspect_terms(tokens_list)
    aspect_polarity = get_polarity(polarity_model, sentence, aspect_list_sent)
    aspect_category = get_category(category_model, sentence, aspect_list_sent)
    return aspect_list_sent, aspect_polarity, aspect_category

def combine_aspect_terms(token_list):
    combined_terms = []
    current_term = ""
    current_tag = ""

    for token_dict in token_list:
        token, tag = list(token_dict.items())[0]

        if tag == "B-ASPECT":
            if current_term:  # Save the current term before starting a new one
                combined_terms.append(current_term.strip())
            current_term = token + " "
            current_tag = "token"
        elif tag == "I-ASPECT" and current_tag == "token":
            current_term += token + " "
        else:
            if current_term:
                combined_terms.append(current_term.strip())
            current_term = ""
            current_tag = ""

    # Add the last term if there's any
    if current_term:
        combined_terms.append(current_term.strip())

    return combined_terms

# Call the function
data, aspect_term_cat_list = extract_positions(root)
data["sent_id"] = pd.to_numeric(data["sent_id"])

aspect_model  = NERModel(
    "distilbert",
    "./combined_ner",
ignore_mismatched_sizes=True, use_cuda=False, args={"slient": True}
)
polarity_model = ClassificationModel(
    "roberta",
    "./polarity", ignore_mismatched_sizes=True, use_cuda=False,
    args={"slient": True},
)

category_model  = ClassificationModel(
    "roberta",
    "./category", ignore_mismatched_sizes=True, use_cuda=False,
    args={"slient": True}
)


unique_sentences = data.text.unique()

from tqdm import tqdm
prediction_data = []
import json
for sent in tqdm(unique_sentences):
    aspect_list_sent, aspect_polarity, aspect_category = gets_output_on_sentence(sent)
    for ent, pol, cat in zip(aspect_list_sent, aspect_polarity, aspect_category):

        if os.path.exists("./ABSA_prediction_data_local.json"):
            prediction_data = json.load(open("./ABSA_prediction_data_local.json", "r"))
            prediction_data.append((sent, ent, pol, cat))
            json.dump(prediction_data, open("./ABSA_prediction_data_local.json", "w"))
        else:
            prediction_data.append((sent, ent, pol, cat))
            json.dump(prediction_data, open("./ABSA_prediction_data_local.json", "w"))

json.dump(prediction_data, open("./ABSA_prediction_data_final.json", "w"))